from simple_flashinfer.env import args, logging
from fastapi import FastAPI, Request, Response
from fastapi import HTTPException
from sse_starlette import EventSourceResponse
from transformers import AttentionInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
from .manager import AutoKVCacheManager
from .parameters import ChatCompletionForm, CompletionForm
from .utils import (
    logits_to_probs,
    block_diagonal_concat_inverted,
)
from tqdm import tqdm
from contextlib import nullcontext
import torch
import json
import asyncio
import flashinfer
import uvicorn
import time
import uuid
import traceback
import os

@torch.compiler.disable
def flashinfer_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    """
    For prefilling, it will pass flashinfer.BatchPrefillWithPagedKVCacheWrapper
    For step decoding, it will pass flashinfer.BatchDecodeWithPagedKVCacheWrapper

    The shape should be,
    query: [1, H, L, D]
    key: [1, H, L, D]
    value: [1, H, L, D]

    While flashinfer input is [L, H, D]
    """
    wrapper = kwargs.get('wrapper')
    manager = kwargs.get('manager')
    prefill = kwargs.get('prefill')
    append_indptr = kwargs.get('append_indptr')
    
    if args.need_autocast:
        query = query.to(args.torch_dtype)
        key = key.to(args.torch_dtype)
        value = value.to(args.torch_dtype)

    query = query[0].transpose(0, 1)
    key = key[0].transpose(0, 1)
    value = value[0].transpose(0, 1)

    layer_attr = 'prefill_layer_idx' if prefill else 'decode_layer_idx'
    layer_idx = getattr(manager, layer_attr)

    batch_attr = 'prefill_batch_ids' if prefill else 'decode_batch_ids'
    batch_ids = getattr(manager, batch_attr)

    manager.append_paged_kv_cache(batch_ids, key, value, append_indptr, layer_idx)
    o = wrapper.run(query, manager.kv_cache[layer_idx])

    if args.compare_sdpa_prefill and prefill:
        diff = torch.diff(append_indptr)
        masks = []
        for l in diff:
            masks.append(torch.tril(torch.ones(l, l)))
            
        masks = block_diagonal_concat_inverted(*masks, dtype = query.dtype).cuda()
        q = query.transpose(0, 1)[None]
        k = key.transpose(0, 1)[None]
        v = value.transpose(0, 1)[None]
        enable_gqa = q.shape[1] != k.shape[1]
        output_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa)
        output_sdpa = output_sdpa[0].transpose(0, 1)
        mean_abs_diff = (output_sdpa - o).abs().mean()
        allclose = torch.allclose(output_sdpa, o, atol=0.125, rtol=0)
        logging.info(f'{layer_idx}, mean abs diff: {mean_abs_diff}, torch.allclose: {allclose}')
        o = output_sdpa

    setattr(manager, layer_attr, layer_idx + 1)
    o = o[None]

    """
    Output shape should be,
    [1, L, H, D]
    """
    if args.need_autocast:
        o = o.to(args.model_dtype)
    return o, None

def load_model():
    global tokenizer, model, manager
    global num_layers, num_heads, num_key_value_heads, head_dim, vocab_size, eos_token_id
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="flashinfer_attention", 
        torch_dtype = args.model_dtype).eval().cuda()
    eos_token_id = model.generation_config.eos_token_id
    if not isinstance(eos_token_id, list):
        eos_token_id = [eos_token_id]
    eos_token_id = torch.tensor(eos_token_id).cuda()
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    vocab_size = config.vocab_size
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads // config.num_key_value_heads)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads)
    manager = AutoKVCacheManager(
        num_layers, 
        num_key_value_heads, 
        head_dim, 
        dtype=args.torch_dtype,
        mem_utilization=args.memory_utilization,
        vocab_size=vocab_size,
        seq_lens=args.max_sequence,
    )

class CUDAGraphDecodeWrapper:
    def __init__(self, decode_fn):
        self.decode_fn = decode_fn
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}

    def warmup(self, key, **inputs):
        for k, v in inputs.items():
            inputs[k] = v.contiguous()
        self.static_inputs[key] = {k: v.clone().cuda() for k, v in inputs.items()}
        self.static_outputs[key] = {}

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.static_outputs[key] = self.decode_fn(**self.static_inputs[key])
        self.graphs[key] = g
    
    def run(self, key, **new_inputs):
        for k in new_inputs:
            if isinstance(new_inputs[k], torch.Tensor):
                self.static_inputs[key][k].copy_(new_inputs[k])
        self.graphs[key].replay()
        return self.static_outputs[key]

def decode(*args, **kwargs):
    return model(*args, **kwargs)

tokenizer = None
model = None
manager = None
num_layers = None
num_heads = None
num_key_value_heads = None
head_dim = None
vocab_size = None
eos_token_id = None
AttentionInterface.register("flashinfer_attention", flashinfer_attention)
workspace_buffer_prefill = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer_prefill, "NHD")
workspace_buffer_decode = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_decode, "NHD")
empty_length = torch.tensor([0]).cuda()
decode_length = torch.tensor([1]).cuda()
prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

if args.torch_profiling:
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    )
else:
    profiler = nullcontext()

app = FastAPI()

@app.middleware("http")
async def add_request_id_and_time(request: Request, call_next):
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
                manager.free(request.state.request_id)
                logging.info(f'freeing kv cache from {request.state.request_id}')
                logging.info(f"{request_id} completed in {duration:.4f} seconds")
                total_token = getattr(request.state, 'total_token', None)
                if total_token is not None:
                    tps = total_token / duration
                    logging.info(f"{request_id}, total token: {total_token}, TPS: {tps:.4f}")

        response.body_iterator = streaming_wrapper()
    else:
        duration = time.perf_counter() - start_time
        manager.free(request.state.request_id)
        logging.info(f'freeing kv cache from {request.state.request_id}')
        logging.info(f"{request_id} completed in {duration:.4f} seconds")
        total_token = getattr(request.state, 'total_token', None)
        if total_token is not None:
            tps = total_token / duration
            logging.info(f"{request_id}, total token: {total_token}, TPS: {tps:.4f}")

    if exception is not None:
        raise exception

    return response

async def process_queue(queue, wrapper, prefill):
    global_step = 0
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.microsleep)

        need_sleep = True
        batch = []

        while not queue.empty():
            try:
                request = await asyncio.wait_for(queue.get(), timeout=1e-6)
                batch.append(request)
                if len(batch) >= args.max_sequence:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break

        if not len(batch):
            continue

        with profiler as prof:
            futures, inputs, position_ids, uuids = zip(*[(b[0], b[1], b[2], b[3]) for b in batch])
            temperature, top_k, top_p, lengths = zip(*[(b[4], b[5], b[6], b[7]) for b in batch])
            lengths_cpu = [inp.shape[0] for inp in inputs]

            try:
                position_ids = (
                    torch.cat([torch.arange(l) for l in lengths_cpu])
                    if prefill
                    else torch.tensor(position_ids)
                )[None].cuda()

                for no, l in enumerate(lengths_cpu):
                    if prefill:
                        manager.allocate(uuids[no], l)
                    else:
                        manager.append_tokens(uuids[no], l)
                    
                input_ids = torch.concat(inputs)[None]
                lengths = torch.concat([empty_length] + list(lengths))
                append_indptr = torch.cumsum(lengths, dim=-1).to(torch.int32)

                kv_indices, kv_indptr, kv_last_page_len = manager.get_append_metadata(uuids)
                if prefill:
                    wrapper.plan(
                        append_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        num_heads,
                        num_key_value_heads,
                        head_dim,
                        manager.block_size,
                        causal=True,
                        q_data_type=args.torch_dtype,
                    )
                else:
                    wrapper.plan(
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        num_heads,
                        num_key_value_heads,
                        head_dim,
                        manager.block_size,
                        pos_encoding_mode="NONE",
                        q_data_type=args.torch_dtype,
                    )
                setattr(manager, "prefill_layer_idx" if prefill else "decode_layer_idx", 0)
                setattr(manager, "prefill_batch_ids" if prefill else "decode_batch_ids", uuids)

                forward = model if prefill else decode
                output = forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                    wrapper=wrapper,
                    manager=manager,
                    prefill=prefill,
                    append_indptr=append_indptr,
                )
                logits = output.logits[0, append_indptr[1:] - 1]
                temperature = torch.concat(temperature)[None].T
                top_k = torch.concat(top_k)
                top_p = torch.concat(top_p)

                mask_penalty = []
                for uuid in uuids:
                    mask_penalty.append(manager.mask_penalty[manager.batch_to_seq_len[uuid]])
                mask_penalty = torch.stack(mask_penalty)

                logits = logits / mask_penalty
                logits = logits / temperature

                idx_next = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                    logits, top_k=top_k, top_p=top_p, deterministic=True,
                )[None].T

                tokens = tokenizer.batch_decode(idx_next)

                for i, fut in enumerate(futures):
                    fut.set_result((idx_next[i], tokens[i]))

            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
        
        if args.torch_profiling:
            try:
                mode = 'prefill' if prefill else 'decode'
                prof.export_chrome_trace(f'{mode}-{global_step}.json')
            except Exception as e:
                print(e)
        
        global_step += 1

async def prefill():
    await process_queue(prefill_queue, prefill_wrapper, prefill=True)

async def step():
    await process_queue(step_queue, decode_wrapper, prefill=False)
    
async def stream(inputs, created, form, request):
    uuid = request.state.request_id
    initial_length = inputs.shape[0]
    inputs = inputs.cuda()

    temperature = max(1e-5, form.temperature)
    temperature = torch.tensor([temperature]).cuda()

    top_k = vocab_size if form.top_k == 0 else form.top_k
    top_k = torch.tensor([top_k]).to(torch.int32).cuda()

    top_p = 1.0 if form.top_p == 0 else form.top_p
    top_p = torch.tensor([top_p]).to(torch.float32).cuda()

    prefill_l = torch.tensor([initial_length]).cuda()

    repetition_penalty = max(1e-5, form.repetition_penalty)
    repetition_penalty_cuda = torch.tensor(repetition_penalty).cuda()
    
    for k in range(form.max_tokens):
        is_disconnected = await request.is_disconnected()
        if is_disconnected:
            break
            
        if k == 0:
            q = prefill_queue
            length = prefill_l
        else:
            q = step_queue
            length = decode_length

        l = k + initial_length
        future = asyncio.Future()
        await q.put((future, inputs, l, uuid, temperature, top_k, top_p, length))
        out = await future
        idx_next, token = out

        if repetition_penalty > 1:
            manager.mask_penalty[manager.batch_to_seq_len[uuid], idx_next[0]] /= repetition_penalty_cuda
        else:
            manager.mask_penalty[manager.batch_to_seq_len[uuid], idx_next[0]] *= repetition_penalty_cuda

        if k == 0:
            request.state.time_first_token = time.time()
        
        if not form.ignore_eos and idx_next[0] in eos_token_id:
            break

        if args.torch_compile:
            """
            I got weird overflow if not clone, like (tensor([5256919935786303302], device='cuda:0'),)
            This will hit CUDA indexing assertion.
            """
            idx_next = idx_next.clone()

        inputs = idx_next

        yield token
        await asyncio.sleep(0)
    
    request.state.total_token = k + initial_length

@app.get('/')
async def index(request: Request = None):
    return {'message': 'hello'}

@app.get('/kv_cache')
async def index(request: Request = None):
    total_kv_cache = manager.max_blocks * manager.block_size
    free_kv_cache = len(manager.free_blocks) * manager.block_size
    utilized_kv_cache = total_kv_cache - free_kv_cache
    return {
        'total_kv_cache': total_kv_cache,
        'free_kv_cache': free_kv_cache,
        'utilized_kv_cache': utilized_kv_cache,
    }

async def handle_stream_response(func, created, request_id, stream_type="completion"):
    async def generator():
        async for data in func:
            if not isinstance(data, str):
                continue
            
            if stream_type == "chat":
                payload = {
                    'id': request_id,
                    'choices': [
                        {
                            'delta': {
                                'content': data,
                                'function_call': None,
                                'role': None,
                                'tool_calls': None
                            },
                            'finish_reason': None,
                            'index': 0,
                            'logprobs': None
                        }
                    ],
                    'created': created,
                    'model': 'model',
                    'object': 'chat.completion.chunk',
                    'system_fingerprint': None
                }
            else:
                payload = {
                    'id': request_id,
                    'choices': [
                        {
                            'text': data,
                            'finish_reason': None,
                            'index': 0,
                            'logprobs': None
                        }
                    ],
                    'created': created,
                    'model': 'model',
                    'object': 'text_completion',
                    'system_fingerprint': None
                }
            yield json.dumps(payload)
            await asyncio.sleep(0)
    return generator()

async def handle_non_stream_response(func, inputs, created, request_id, stream_type="completion"):
    tokens = []
    async for data in func:
        if isinstance(data, str):
            tokens.append(data)

    output_text = ''.join(tokens)
    base = {
        'id': request_id,
        'created': created,
        'model': 'model',
        'system_fingerprint': None,
        'usage': {
            'completion_tokens': len(tokens),
            'prompt_tokens': len(inputs),
            'total_tokens': len(inputs) + len(tokens),
        }
    }

    if stream_type == "chat":
        base.update({
            'object': 'chat.completion',
            'choices': [
                {
                    'finish_reason': 'stop',
                    'index': 0,
                    'logprobs': None,
                    'message': {
                        'content': output_text,
                        'role': 'assistant',
                        'function_call': None,
                        'tool_calls': None
                    }
                }
            ]
        })
    else:
        base.update({
            'object': 'text_completion',
            'choices': [
                {
                    'finish_reason': 'stop',
                    'index': 0,
                    'logprobs': None,
                    'text': output_text,
                }
            ]
        })
    return base

async def handle_completion(form, request, tokenizer, is_chat=False):
    created = int(time.time())
    request_id = request.state.request_id

    if is_chat:
        prompt = tokenizer.apply_chat_template(form.messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = form.prompt

    inputs = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)[0]

    func = stream(inputs=inputs, created=created, form=form, request=request)
    stream_type = "chat" if is_chat else "completion"

    if form.stream:
        return EventSourceResponse(
            await handle_stream_response(func, created, request_id, stream_type),
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
    else:
        return await handle_non_stream_response(func, inputs, created, request_id, stream_type)

@app.post('/completions')
async def completions_main(form: CompletionForm, request: Request = None):
    return await handle_completion(form, request, tokenizer, is_chat=False)

@app.post('/chat/completions')
async def chat_completions_main(form: ChatCompletionForm, request: Request = None):
    return await handle_completion(form, request, tokenizer, is_chat=True)

@app.on_event("startup")
async def startup_event():
    load_model()
    app.state.background_prefill = asyncio.create_task(prefill())
    app.state.background_step = asyncio.create_task(step())

    logging.info('warming up')
    
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
        r = await chat_completions_main(form=form, request=request)
        manager.free(request.state.request_id)
    
    if args.torch_compile:
        global decode

        decode = torch.compile(decode, mode=args.torch_compile_mode, dynamic=True)
        for i in tqdm(range(args.max_sequence), desc='warming up torch compile'):
            tasks = []
            for k in range(i + 1):
                request = Request(dummy_scope.copy(), receive=receive)
                request.state.request_id = f'dummy-{k}'
                task = asyncio.create_task(chat_completions_main(form=form, request=request))
                tasks.append(task)
            
            await asyncio.gather(*tasks)

            for k in range(i + 1):
                manager.free(f'dummy-{k}')
    
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        loop="uvloop",
    )