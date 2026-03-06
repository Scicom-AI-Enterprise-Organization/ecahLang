from ecahlang.env import args, logging
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
def ecah_attention(
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

    if manager.cuda_graph_mode and not prefill:
        bucket_size = query.shape[0]
        manager.append_paged_kv_cache_cuda_graph(key, value, bucket_size, layer_idx)
    else:
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
        args.model, attn_implementation="ecah_attention",
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
    def __init__(self, decode_and_sample_fn):
        self.fn = decode_and_sample_fn
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}

    def warmup(self, bucket_size, capture_stream, **inputs):
        for k, v in inputs.items():
            inputs[k] = v.contiguous()
        self.static_inputs[bucket_size] = {k: v.clone().cuda() for k, v in inputs.items()}

        for _ in range(2):
            self.fn(**self.static_inputs[bucket_size])

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream):
            out = self.fn(**self.static_inputs[bucket_size])
        self.static_outputs[bucket_size] = out
        self.graphs[bucket_size] = g

    def run(self, bucket_size, **new_inputs):
        for k, v in new_inputs.items():
            if isinstance(v, torch.Tensor):
                self.static_inputs[bucket_size][k].copy_(v)
        self.graphs[bucket_size].replay()
        return self.static_outputs[bucket_size]

def decode(*args, **kwargs):
    return model(*args, **kwargs)

def decode_forward(input_ids, position_ids, append_indptr, mask_penalty, temperature):
    manager.decode_layer_idx = 0
    output = decode(
        input_ids=input_ids, position_ids=position_ids, use_cache=False,
        wrapper=decode_wrapper, manager=manager, prefill=False, append_indptr=append_indptr,
    )
    logits = output.logits[0]
    logits = logits / mask_penalty
    logits = logits / temperature
    return logits

def get_bucket_sizes(max_sequence):
    sizes = []
    s = 1
    while s <= max_sequence:
        sizes.append(s)
        s *= 2
    if sizes[-1] < max_sequence:
        sizes.append(max_sequence)
    return sizes

def next_bucket(n, bucket_sizes):
    for bs in bucket_sizes:
        if bs >= n:
            return bs
    return bucket_sizes[-1]

tokenizer = None
model = None
manager = None
num_layers = None
num_heads = None
num_key_value_heads = None
head_dim = None
vocab_size = None
eos_token_id = None
cuda_graph_runner = None
bucket_sizes = []
AttentionInterface.register("ecah_attention", ecah_attention)
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
    fwd_stream = torch.cuda.Stream()
    global_step = 0
    need_sleep = True
    next_batch = None  # Phase 2: pre-collected batch buffer

    while True:
        # Phase 2: use pre-collected batch if available, otherwise collect fresh
        if next_batch is not None:
            batch = next_batch
            next_batch = None
        else:
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
            rep_penalties = [b[8] if len(b) > 8 else 1.0 for b in batch]
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

                if (args.cuda_graph or args.torch_compile) and not prefill and bucket_sizes:
                    # ====== BUCKETED DECODE PATH (CUDA graph or torch_compile) ======
                    n = len(uuids)
                    bucket = next_bucket(n, bucket_sizes)

                    manager.fill_cuda_graph_metadata(bucket, uuids)

                    decode_wrapper.plan(
                        manager._cg_kv_indptr[bucket],
                        manager._cg_kv_indices[bucket],
                        manager._cg_kv_last_page_len[bucket],
                        num_heads, num_key_value_heads, head_dim, manager.block_size,
                        pos_encoding_mode="NONE", q_data_type=args.torch_dtype,
                    )

                    real_ids = torch.concat(inputs).to(torch.long)
                    padded_ids = torch.zeros(bucket, dtype=torch.long, device="cuda")
                    padded_ids[:n] = real_ids

                    padded_pos = torch.zeros(bucket, dtype=torch.long, device="cuda")
                    padded_pos[:n] = position_ids[0]

                    padded_mask = torch.ones(bucket, vocab_size, dtype=args.torch_dtype, device="cuda")
                    for i, u in enumerate(uuids):
                        padded_mask[i] = manager.mask_penalty[manager.batch_to_seq_len[u]]

                    manager.fill_cuda_graph_sampling_params(bucket, n, temperature, top_k, top_p)

                    manager.decode_layer_idx = 0

                    if args.cuda_graph:
                        manager.cuda_graph_mode = True

                        fwd_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(fwd_stream):
                            logits = cuda_graph_runner.run(
                                bucket,
                                input_ids=padded_ids[None],
                                position_ids=padded_pos[None],
                                append_indptr=manager._cg_append_indptr[bucket],
                                mask_penalty=padded_mask,
                                temperature=manager._cg_temperature[bucket],
                            )
                            idx_next_full = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                logits, top_k=manager._cg_top_k[bucket], top_p=manager._cg_top_p[bucket], deterministic=True,
                            )

                        manager.cuda_graph_mode = False
                    else:
                        # torch_compile bucketed decode path
                        manager.decode_batch_ids = tuple(list(uuids) + [uuids[0]] * (bucket - n))

                        fwd_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(fwd_stream):
                            output = decode(
                                input_ids=padded_ids[None],
                                position_ids=padded_pos[None],
                                use_cache=False,
                                wrapper=decode_wrapper,
                                manager=manager,
                                prefill=False,
                                append_indptr=manager._cg_append_indptr[bucket],
                            )
                            logits = output.logits[0]
                            logits = logits / padded_mask
                            logits = logits / manager._cg_temperature[bucket]
                            idx_next_full = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                logits, top_k=manager._cg_top_k[bucket], top_p=manager._cg_top_p[bucket], deterministic=True,
                            )

                    await asyncio.sleep(0)
                    next_batch = []
                    while not queue.empty() and len(next_batch) < args.max_sequence:
                        try:
                            request = await asyncio.wait_for(queue.get(), timeout=1e-6)
                            next_batch.append(request)
                        except asyncio.TimeoutError:
                            break
                    if not next_batch:
                        next_batch = None

                    fwd_stream.synchronize()

                    idx_next = idx_next_full[:n].unsqueeze(1)
                    loop = asyncio.get_running_loop()
                    idx_next_cpu = idx_next.cpu()
                    tokens = await loop.run_in_executor(None, tokenizer.batch_decode, idx_next_cpu)
                    for i, fut in enumerate(futures):
                        fut.set_result((idx_next[i], tokens[i]))

                else:
                    # ====== ORIGINAL PATH (prefill or non-cuda-graph decode) ======
                    input_ids = torch.concat(inputs)[None]
                    lengths = torch.tensor([0] + list(lengths), device="cuda")
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

                    n = len(uuids)
                    manager.fill_sampling_params(n, temperature, top_k, top_p)

                    # Overlap: ensure fwd_stream waits for prior default-stream work (wrapper.plan)
                    fwd_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(fwd_stream):
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
                        temperature_t = manager.sampling_temperature_gpu[:n]
                        top_k_t = manager.sampling_top_k_gpu[:n]
                        top_p_t = manager.sampling_top_p_gpu[:n]

                        mask_penalty = []
                        for u in uuids:
                            mask_penalty.append(manager.mask_penalty[manager.batch_to_seq_len[u]])
                        mask_penalty = torch.stack(mask_penalty)

                        logits = logits / mask_penalty
                        logits = logits / temperature_t

                        idx_next = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                            logits, top_k=top_k_t, top_p=top_p_t, deterministic=True,
                        )[None].T

                    # Phase 2: yield to event loop while GPU still running
                    await asyncio.sleep(0)

                    # Phase 2: pre-collect next batch while GPU finishes
                    next_batch = []
                    while not queue.empty() and len(next_batch) < args.max_sequence:
                        try:
                            request = await asyncio.wait_for(queue.get(), timeout=1e-6)
                            next_batch.append(request)
                        except asyncio.TimeoutError:
                            break
                    if not next_batch:
                        next_batch = None

                    fwd_stream.synchronize()

                    # Phase 3: multi-step decode loop
                    num_steps = args.multi_step if not prefill else 1
                    if num_steps > 1:
                        accumulated = [[(idx_next[i], None, idx_next[i][0] in eos_token_id)] for i in range(len(uuids))]
                        active_mask = [not (idx_next[i][0] in eos_token_id) for i in range(len(uuids))]
                        position_ids_track = [position_ids[0, i].item() + 1 for i in range(len(uuids))]

                        # apply repetition penalty for step 0
                        for i in range(len(uuids)):
                            if rep_penalties[i] > 1:
                                manager.mask_penalty[manager.batch_to_seq_len[uuids[i]], idx_next[i][0]] /= rep_penalties[i]

                        for step in range(1, num_steps):
                            active_indices = [i for i, a in enumerate(active_mask) if a]
                            if not active_indices:
                                break

                            step_input_ids = torch.stack([idx_next[i] for i in active_indices]).T  # [1, active_batch]
                            step_position_ids = torch.tensor([position_ids_track[i] for i in active_indices])[None].cuda()

                            for i in active_indices:
                                manager.append_tokens(uuids[i], 1)

                            active_uuids = tuple(uuids[i] for i in active_indices)
                            kv_indices, kv_indptr, kv_last_page_len = manager.get_append_metadata(active_uuids)
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

                            setattr(manager, "decode_layer_idx", 0)
                            setattr(manager, "decode_batch_ids", active_uuids)

                            fwd_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(fwd_stream):
                                step_output = forward(
                                    input_ids=step_input_ids,
                                    position_ids=step_position_ids,
                                    use_cache=False,
                                    wrapper=wrapper,
                                    manager=manager,
                                    prefill=False,
                                    append_indptr=append_indptr[:len(active_indices) + 1] if len(active_indices) < len(uuids) else append_indptr,
                                )
                                step_logits = step_output.logits[0]

                                step_mask_penalty = []
                                for i in active_indices:
                                    step_mask_penalty.append(manager.mask_penalty[manager.batch_to_seq_len[uuids[i]]])
                                step_mask_penalty = torch.stack(step_mask_penalty)

                                active_temps = [temperature[i] for i in active_indices]
                                active_top_ks = [top_k[i] for i in active_indices]
                                active_top_ps = [top_p[i] for i in active_indices]
                                manager.fill_sampling_params(len(active_indices), active_temps, active_top_ks, active_top_ps)
                                step_temp = manager.sampling_temperature_gpu[:len(active_indices)]
                                step_top_k = manager.sampling_top_k_gpu[:len(active_indices)]
                                step_top_p = manager.sampling_top_p_gpu[:len(active_indices)]

                                step_logits = step_logits / step_mask_penalty
                                step_logits = step_logits / step_temp

                                step_idx_next = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                                    step_logits, top_k=step_top_k, top_p=step_top_p, deterministic=True,
                                )[None].T

                            fwd_stream.synchronize()

                            # scatter back into idx_next and accumulate
                            for local_idx, global_idx in enumerate(active_indices):
                                token_id = step_idx_next[local_idx]
                                hit_eos = token_id[0] in eos_token_id
                                idx_next[global_idx] = token_id
                                accumulated[global_idx].append((token_id, None, hit_eos))
                                if hit_eos:
                                    active_mask[global_idx] = False
                                if rep_penalties[global_idx] > 1:
                                    manager.mask_penalty[manager.batch_to_seq_len[uuids[global_idx]], token_id[0]] /= rep_penalties[global_idx]
                                position_ids_track[global_idx] += 1

                    if num_steps > 1:
                        loop = asyncio.get_running_loop()
                        for i in range(len(uuids)):
                            all_token_ids = torch.stack([t[0] for t in accumulated[i]])
                            decoded = await loop.run_in_executor(None, tokenizer.batch_decode, all_token_ids)
                            result = [(accumulated[i][j][0], decoded[j], accumulated[i][j][2]) for j in range(len(accumulated[i]))]
                            futures[i].set_result(result)
                    else:
                        # Single step path (prefill or multi_step=1)
                        # Phase 1: async detokenization
                        loop = asyncio.get_running_loop()
                        idx_next_cpu = idx_next.cpu()
                        tokens = await loop.run_in_executor(None, tokenizer.batch_decode, idx_next_cpu)

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
    top_k = vocab_size if form.top_k == 0 else form.top_k
    top_p = 1.0 if form.top_p == 0 else form.top_p
    repetition_penalty = max(1e-5, form.repetition_penalty)

    k = 0
    while k < form.max_tokens:
        is_disconnected = await request.is_disconnected()
        if is_disconnected:
            break

        if k == 0:
            q = prefill_queue
            length = initial_length
        else:
            q = step_queue
            length = 1

        l = k + initial_length
        future = asyncio.Future()
        await q.put((future, inputs, l, uuid, temperature, top_k, top_p, length, repetition_penalty))
        out = await future

        if isinstance(out, list):
            # Phase 3: multi-step response — list of (idx_next, token, hit_eos)
            hit_eos_outer = False
            for step_i, (step_idx_next, step_token, hit_eos) in enumerate(out):
                if k == 0 and step_i == 0:
                    request.state.time_first_token = time.time()

                # repetition penalty already applied in process_queue for multi-step
                if not form.ignore_eos and hit_eos:
                    hit_eos_outer = True
                    break

                if args.torch_compile or args.cuda_graph:
                    step_idx_next = step_idx_next.clone()

                inputs = step_idx_next
                yield step_token
                await asyncio.sleep(0)

            k += len(out) if not hit_eos_outer else step_i + 1
            if hit_eos_outer:
                break
        else:
            # Single step (prefill or multi_step=1)
            idx_next, token = out

            if repetition_penalty > 1:
                manager.mask_penalty[manager.batch_to_seq_len[uuid], idx_next[0]] /= repetition_penalty
            else:
                manager.mask_penalty[manager.batch_to_seq_len[uuid], idx_next[0]] *= repetition_penalty

            if k == 0:
                request.state.time_first_token = time.time()

            if not form.ignore_eos and idx_next[0] in eos_token_id:
                break

            if args.torch_compile or args.cuda_graph:
                """
                I got weird overflow if not clone, like (tensor([5256919935786303302], device='cuda:0'),)
                This will hit CUDA indexing assertion.
                CUDA graph has the same issue: static output tensor gets overwritten on next replay.
                """
                idx_next = idx_next.clone()

            inputs = idx_next

            yield token
            await asyncio.sleep(0)
            k += 1

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
    global decode, decode_forward, cuda_graph_runner, bucket_sizes

    load_model()
    manager.init_sampling_buffers(args.max_sequence)
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

    if args.cuda_graph:
        bucket_sizes = get_bucket_sizes(args.max_sequence)
        logging.info(f'CUDA Graph bucket sizes: {bucket_sizes}')

        manager.init_cuda_graph_buffers(bucket_sizes)

        capture_stream = torch.cuda.Stream()
        cuda_graph_runner = CUDAGraphDecodeWrapper(decode_forward)

        for bs in tqdm(bucket_sizes, desc='warming up CUDA Graph'):
            dummy_uuids = []
            for k in range(bs):
                uid = f'cg-warmup-{bs}-{k}'
                dummy_uuids.append(uid)
                manager.allocate(uid, 1)
                manager.append_tokens(uid, 1)

            manager.fill_cuda_graph_metadata(bs, tuple(dummy_uuids))

            decode_wrapper.plan(
                manager._cg_kv_indptr[bs],
                manager._cg_kv_indices[bs],
                manager._cg_kv_last_page_len[bs],
                num_heads, num_key_value_heads, head_dim, manager.block_size,
                pos_encoding_mode="NONE", q_data_type=args.torch_dtype,
            )

            manager.cuda_graph_mode = True
            manager.decode_layer_idx = 0
            manager.decode_batch_ids = tuple(dummy_uuids)

            cuda_graph_runner.warmup(
                bs,
                capture_stream=capture_stream,
                input_ids=torch.zeros(1, bs, dtype=torch.long, device="cuda"),
                position_ids=torch.ones(1, bs, dtype=torch.long, device="cuda"),
                append_indptr=manager._cg_append_indptr[bs],
                mask_penalty=torch.ones(bs, vocab_size, dtype=args.torch_dtype, device="cuda"),
                temperature=manager._cg_temperature[bs],
            )

            manager.cuda_graph_mode = False

            for uid in dummy_uuids:
                manager.free(uid)

        logging.info('CUDA Graph warmup complete')

    elif args.torch_compile:
        bucket_sizes = get_bucket_sizes(args.max_sequence)
        logging.info(f'torch_compile bucket sizes: {bucket_sizes}')

        manager.init_cuda_graph_buffers(bucket_sizes)

        decode = torch.compile(decode, mode=args.torch_compile_mode, dynamic=False)

        for bs in tqdm(bucket_sizes, desc='warming up torch compile'):
            dummy_uuids = []
            for k in range(bs):
                uid = f'tc-warmup-{bs}-{k}'
                dummy_uuids.append(uid)
                manager.allocate(uid, 1)
                manager.append_tokens(uid, 1)

            manager.fill_cuda_graph_metadata(bs, tuple(dummy_uuids))

            decode_wrapper.plan(
                manager._cg_kv_indptr[bs],
                manager._cg_kv_indices[bs],
                manager._cg_kv_last_page_len[bs],
                num_heads, num_key_value_heads, head_dim, manager.block_size,
                pos_encoding_mode="NONE", q_data_type=args.torch_dtype,
            )

            manager.decode_layer_idx = 0
            manager.decode_batch_ids = tuple(dummy_uuids)

            decode(
                input_ids=torch.zeros(1, bs, dtype=torch.long, device="cuda"),
                position_ids=torch.ones(1, bs, dtype=torch.long, device="cuda"),
                use_cache=False,
                wrapper=decode_wrapper,
                manager=manager,
                prefill=False,
                append_indptr=manager._cg_append_indptr[bs],
            )

            for uid in dummy_uuids:
                manager.free(uid)

        logging.info('torch compile warmup complete')

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
        loop="uvloop",
    )
