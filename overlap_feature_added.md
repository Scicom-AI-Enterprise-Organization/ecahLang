# Overlap Scheduler — Feature Added

## What Was Added

A **CUDA stream-based overlap scheduler** in `ecahlang/main.py`. Three lines of logic added to `process_queue()` — existing code just re-indented, nothing removed.

## Why

Previously, GPU and CPU work were fully sequential:

```
collect batch (CPU) → prepare (CPU) → GPU forward → GPU sample → batch_decode (CPU) → set_result (CPU)
                                       GPU busy      GPU busy
                                       CPU idle      CPU idle
```

GPU idles while CPU works. CPU idles while GPU works. No overlap.

## The Change

Each `process_queue` (prefill and decode) now gets its own CUDA stream. GPU work (forward + sampling) runs on that stream, then we **yield to the asyncio event loop** while the GPU is still computing. This lets the other queue do CPU work (collect batch, prepare tensors, plan FlashInfer) concurrently.

### Code Added (3 things)

**1. Create stream** (top of `process_queue`):
```python
fwd_stream = torch.cuda.Stream()
```

**2. Wrap forward + sampling in stream context**:
```python
fwd_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(fwd_stream):
    output = forward(...)
    # ... sampling (logits, mask_penalty, top_k_top_p) ...
    idx_next = flashinfer.sampling.top_k_top_p_sampling_from_logits(...)
```

**3. Yield + sync before using results**:
```python
await asyncio.sleep(0)          # yield to event loop while GPU runs
fwd_stream.synchronize()        # wait for GPU results
tokens = tokenizer.batch_decode(idx_next)  # now safe to use
```

## How It Works

```
Decode queue:  [prepare] → [launch on fwd_stream] → [YIELD] ............→ [sync] → [batch_decode]
                                                        ↓                    ↑
                                                   event loop runs      GPU done
                                                        ↓
Prefill queue: ........................................[prepare] → [launch] → [YIELD] → ...
                                                        ↑
                                              runs while decode GPU busy
```

`fwd_stream.wait_stream(current_stream)` ensures the forward stream waits for prior default-stream work (`wrapper.plan()`). The `with torch.cuda.stream(fwd_stream)` routes all GPU ops to the forward stream. `await asyncio.sleep(0)` yields control so other asyncio coroutines can run. `fwd_stream.synchronize()` blocks until GPU results are ready.

## Benchmark Results

**Model**: Qwen2.5-3B-Instruct (FP16) on RTX 3090  
**Mode**: torch.compile reduce-overhead, max_sequence=11  
**Benchmark**: 384 tokens per request, ignore_eos=True

### reduce-overhead (before — no overlap)

| Concurrency | First Token (s) | Total Response (s) | Throughput |
|---|---|---|---|
| 1 | 0.106 | 22.37 | 17.2 tok/s |
| 5 | 0.184 | 45.89 | 41.8 tok/s |
| 10 | 0.141 | 47.69 | 80.5 tok/s |

### overlap-scheduler-1.0 (after — with CUDA stream)

| Concurrency | First Token (s) | Total Response (s) | Throughput |
|---|---|---|---|
| 1 | 0.092 (-13%) | 21.57 (-3.6%) | 17.8 tok/s |
| 5 | 0.145 (-21%) | 45.54 (-0.8%) | 42.2 tok/s |
| 10 | 0.114 (-19%) | **24.60 (-48%)** | **156.1 tok/s** |

### Key Takeaway

At **c10** (near max_sequence=11), the overlap scheduler is **~2x faster**. Both prefill and decode queues are busy, so the yield point lets one queues CPU work overlap with the others GPU execution.

First token latency improved 13-21% across all concurrency levels.

## What This Does NOT Change

- No multi-process architecture (tokenizer/detokenizer still synchronous)
- No future tokens or double buffering
- No changes to KV cache manager, attention, model loading, or API endpoints
- No new CLI flags — overlap is always on
- Existing code logic unchanged — just wrapped in stream context
