"""
FastAPI server: endpoints, middleware, startup/shutdown.

Provides OpenAI-compatible API:
  POST /completions          - text completion
  POST /chat/completions     - chat completion
  GET  /                     - health check
  GET  /kv_cache             - KV cache utilization stats

Inspired by:
- vLLM entrypoints/api_server.py
- SGLang entrypoints/
"""

import asyncio
import logging
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from sse_starlette import EventSourceResponse
from tqdm import tqdm

from ..server_args import args
from ..entrypoints.protocol import ChatCompletionForm, CompletionForm
from ..core.request_state import RequestState
from ..core.scheduler import Scheduler
from ..mem.paged_kv_manager import PagedKVCacheManager
from ..model_executor.model_runner import ModelRunner
from .streaming import stream_tokens, format_sse_stream, collect_non_stream_response

logger = logging.getLogger(__name__)

# Global instances (initialized at startup)
model_runner = ModelRunner(args)
kv_manager = None
scheduler = None

app = FastAPI()


@app.middleware("http")
async def request_lifecycle(request: Request, call_next):
    """Assign request ID, track timing, clean up KV cache on completion."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    exception = None

    try:
        response = await call_next(request)
    except Exception as e:
        exception = e
        response = Response("Internal server error", status_code=500)

    if hasattr(response, "body_iterator"):
        original_iterator = response.body_iterator

        async def streaming_wrapper():
            try:
                async for chunk in original_iterator:
                    yield chunk
            finally:
                duration = time.perf_counter() - start_time
                kv_manager.free(request.state.request_id)
                logger.info(f'freed KV cache for {request.state.request_id}')
                logger.info(f'{request_id} completed in {duration:.4f}s')
                total_token = getattr(request.state, 'total_token', None)
                if total_token is not None:
                    tps = total_token / duration
                    logger.info(f'{request_id}, total tokens: {total_token}, TPS: {tps:.4f}')

        response.body_iterator = streaming_wrapper()
    else:
        duration = time.perf_counter() - start_time
        kv_manager.free(request.state.request_id)
        logger.info(f'freed KV cache for {request.state.request_id}')
        logger.info(f'{request_id} completed in {duration:.4f}s')
        total_token = getattr(request.state, 'total_token', None)
        if total_token is not None:
            tps = total_token / duration
            logger.info(f'{request_id}, total tokens: {total_token}, TPS: {tps:.4f}')

    if exception is not None:
        raise exception

    return response


@app.get('/')
async def index():
    return {'message': 'ecahLang'}


@app.get('/kv_cache')
async def kv_cache_stats():
    return kv_manager.get_stats()


async def handle_completion(form, request, is_chat=False):
    """Shared handler for both /completions and /chat/completions."""
    created = int(time.time())
    request_id = request.state.request_id

    if is_chat:
        prompt = model_runner.apply_chat_template(form.messages)
    else:
        prompt = form.prompt

    input_ids = model_runner.tokenize(prompt)

    req_state = RequestState(
        request_id=request_id,
        input_ids=input_ids,
        temperature=form.temperature,
        top_k=form.top_k,
        top_p=form.top_p,
        repetition_penalty=form.repetition_penalty,
        max_tokens=form.max_tokens,
        ignore_eos=form.ignore_eos,
        vocab_size=model_runner.vocab_size,
    )

    token_gen = stream_tokens(scheduler, req_state, request, model_runner.eos_token_id)
    stream_type = "chat" if is_chat else "completion"

    if form.stream:
        return EventSourceResponse(
            format_sse_stream(token_gen, created, request_id, stream_type),
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            },
        )
    else:
        result = await collect_non_stream_response(token_gen, input_ids, created, request_id, stream_type)
        request.state.total_token = req_state.total_tokens
        return result


@app.post('/completions')
async def completions(form: CompletionForm, request: Request = None):
    return await handle_completion(form, request, is_chat=False)


@app.post('/chat/completions')
async def chat_completions(form: ChatCompletionForm, request: Request = None):
    return await handle_completion(form, request, is_chat=True)


@app.on_event("startup")
async def startup_event():
    global kv_manager, scheduler

    # Load model
    model_runner.load()

    # Initialize KV cache manager (auto-sizes from free GPU memory)
    kv_manager = PagedKVCacheManager(
        num_layers=model_runner.num_layers,
        num_kv_heads=model_runner.num_kv_heads,
        head_dim=model_runner.head_dim,
        dtype=args.torch_dtype,
        mem_utilization=args.memory_utilization,
        vocab_size=model_runner.vocab_size,
        seq_lens=args.max_sequence,
    )

    # Initialize scheduler
    scheduler = Scheduler(args, model_runner, kv_manager)
    scheduler.start()

    # Warmup FlashInfer
    logger.info('Warming up FlashInfer...')
    dummy_scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "headers": [],
        "scheme": "http",
        "path": "/",
        "query_string": b"",
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    for _ in tqdm(range(2), desc='warming up FlashInfer'):
        request = Request(dummy_scope.copy(), receive=receive)
        request.state.request_id = 'dummy'
        form = ChatCompletionForm()
        await chat_completions(form=form, request=request)
        kv_manager.free(request.state.request_id)

    # Optional: torch.compile warmup
    if args.torch_compile:
        model_runner.setup_torch_compile()
        logger.info('Warming up torch.compile...')
        for i in tqdm(range(args.max_sequence), desc='warming up torch.compile'):
            tasks = []
            for k in range(i + 1):
                request = Request(dummy_scope.copy(), receive=receive)
                request.state.request_id = f'dummy-{k}'
                task = asyncio.create_task(
                    chat_completions(form=ChatCompletionForm(), request=request)
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            for k in range(i + 1):
                kv_manager.free(f'dummy-{k}')

    logger.info('ecahLang ready to serve')


def run():
    """Run the server."""
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        loop="uvloop",
    )
