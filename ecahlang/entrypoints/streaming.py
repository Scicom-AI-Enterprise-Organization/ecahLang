"""
SSE streaming and per-request stream() coroutine.

Each request spawns a stream() async generator that:
  1. Iteration 0 (prefill): puts full input_ids into prefill_queue, awaits result
  2. Iteration 1..N (decode): puts single token into step_queue, awaits result
  3. Yields token text as SSE events
  4. Stops on EOS or max_tokens

Inspired by:
- vLLM entrypoints/ (streaming responses)
- SGLang managers/detokenizer_manager.py (token streaming)
"""

import asyncio
import json
import time
import torch


async def stream_tokens(scheduler, request_state, request, eos_token_id):
    """
    Async generator that yields decoded token strings one at a time.

    This is the per-request lifecycle: prefill → decode → decode → ... → done.
    """
    uuid = request_state.request_id
    initial_length = request_state.initial_length
    inputs = request_state.current_input
    temperature = request_state.temperature
    top_k = request_state.top_k
    top_p = request_state.top_p
    repetition_penalty = request_state.repetition_penalty
    repetition_penalty_cuda = torch.tensor(repetition_penalty).cuda()

    prefill_l = torch.tensor([initial_length]).cuda()

    for k in range(request_state.max_tokens):
        # Check client disconnect
        is_disconnected = await request.is_disconnected()
        if is_disconnected:
            break

        if k == 0:
            queue_fn = scheduler.enqueue_prefill
            length = prefill_l
        else:
            queue_fn = scheduler.enqueue_step
            length = scheduler.decode_length

        position = k + initial_length
        future = asyncio.Future()
        await queue_fn(future, inputs, position, uuid, temperature, top_k, top_p, length)
        out = await future
        idx_next, token = out

        # Apply repetition penalty
        seq_idx = scheduler.kv_manager.batch_to_seq_len[uuid]
        if repetition_penalty > 1:
            scheduler.kv_manager.mask_penalty[seq_idx, idx_next[0]] /= repetition_penalty_cuda
        else:
            scheduler.kv_manager.mask_penalty[seq_idx, idx_next[0]] *= repetition_penalty_cuda

        if k == 0:
            request_state.time_first_token = time.perf_counter()

        # Check EOS
        if not request_state.ignore_eos and idx_next[0] in eos_token_id:
            break

        # Clone to avoid torch.compile overflow issues
        if scheduler.args.torch_compile:
            idx_next = idx_next.clone()

        inputs = idx_next
        request_state.advance(idx_next)

        yield token
        await asyncio.sleep(0)

    request_state.generated_tokens = k + initial_length


async def format_sse_stream(token_generator, created, request_id, stream_type="completion"):
    """Wrap token generator into SSE-formatted JSON payloads."""
    async for data in token_generator:
        if not isinstance(data, str):
            continue

        if stream_type == "chat":
            payload = {
                'id': request_id,
                'choices': [{
                    'delta': {
                        'content': data,
                        'function_call': None,
                        'role': None,
                        'tool_calls': None,
                    },
                    'finish_reason': None,
                    'index': 0,
                    'logprobs': None,
                }],
                'created': created,
                'model': 'model',
                'object': 'chat.completion.chunk',
                'system_fingerprint': None,
            }
        else:
            payload = {
                'id': request_id,
                'choices': [{
                    'text': data,
                    'finish_reason': None,
                    'index': 0,
                    'logprobs': None,
                }],
                'created': created,
                'model': 'model',
                'object': 'text_completion',
                'system_fingerprint': None,
            }

        yield json.dumps(payload)
        await asyncio.sleep(0)


async def collect_non_stream_response(token_generator, input_ids, created, request_id, stream_type="completion"):
    """Collect all tokens and return a single response object."""
    tokens = []
    async for data in token_generator:
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
            'prompt_tokens': len(input_ids),
            'total_tokens': len(input_ids) + len(tokens),
        },
    }

    if stream_type == "chat":
        base.update({
            'object': 'chat.completion',
            'choices': [{
                'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'message': {
                    'content': output_text,
                    'role': 'assistant',
                    'function_call': None,
                    'tool_calls': None,
                },
            }],
        })
    else:
        base.update({
            'object': 'text_completion',
            'choices': [{
                'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'text': output_text,
            }],
        })

    return base
