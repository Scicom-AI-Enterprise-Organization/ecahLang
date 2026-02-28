# ecahLang

Simple continuous batching CausalLM from HuggingFace Transformer using FlashInfer.

1. Simple paged KV cache manager.
2. Radix tree KV cache manager with prefix sharing.
3. Torch compile support.
4. FP32 support but downcast and upcast attention forward.
5. Support top-k, top-p, temperature and repetition penalty for sampling.
6. Background detokenizer (threaded).
7. CUDA stream overlap for compute/communication.
8. Modular architecture inspired by vLLM and SGLang.

## How to install

Using uv,

```bash
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ecahLang
```

Or you can git clone and install,

```bash
git clone https://github.com/Scicom-AI-Enterprise-Organization/ecahLang && cd ecahLang
uv pip install -e .
```

## How to local

### Supported parameters

```bash
python3 -m ecahlang --help
```

```text
usage: __main__.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--microsleep MICROSLEEP]
               [--max_sequence MAX_SEQUENCE] [--memory_utilization MEMORY_UTILIZATION]
               [--compare-sdpa-prefill COMPARE_SDPA_PREFILL] [--model MODEL] [--torch_dtype TORCH_DTYPE]
               [--torch_dtype_autocast TORCH_DTYPE_AUTOCAST] [--torch_profiling TORCH_PROFILING]
               [--torch_compile TORCH_COMPILE] [--torch_compile_mode TORCH_COMPILE_MODE] [--cuda_graph CUDA_GRAPH]

ecahLang - Continuous Batching LLM Inference

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --microsleep MICROSLEEP
                        microsleep to group batching, 1 / 1e-4 = 10k steps/sec (default: 0.0001, env: MICROSLEEP)
  --max_sequence MAX_SEQUENCE
                        max batch size per prefill or decode step (default: 128, env: MAX_SEQUENCE)
  --memory_utilization MEMORY_UTILIZATION
                        fraction of free GPU memory for KV cache pages (default: 0.9, env: MEMORY_UTILIZATION)
  --compare-sdpa-prefill COMPARE_SDPA_PREFILL
                        compare FlashInfer attention output with SDPA during prefill (default: False, env: COMPARE_SDPA_PREFILL)
  --model MODEL         HuggingFace model name or path (default: meta-llama/Llama-3.2-1B-Instruct, env: MODEL)
  --torch_dtype TORCH_DTYPE
                        model dtype: float16, bfloat16, float32 (default: float16, env: TORCH_DTYPE)
  --torch_dtype_autocast TORCH_DTYPE_AUTOCAST
                        autocast dtype when model is float32 (default: float16, env: TORCH_DTYPE_AUTOCAST)
  --torch_profiling TORCH_PROFILING
                        profile prefill and step with torch profiler (default: False, env: TORCH_PROFILING)
  --torch_compile TORCH_COMPILE
                        torch.compile for decode (default: False, env: TORCH_COMPILE)
  --torch_compile_mode TORCH_COMPILE_MODE
                        torch.compile mode (default: default, env: TORCH_COMPILE_MODE)
  --cuda_graph CUDA_GRAPH
                        capture CUDA Graph for decode (default: False, env: CUDA_GRAPH)
```

**We support both args and OS environment**.

### Run meta-llama/Llama-3.2-1B-Instruct

```bash
python3 -m ecahlang \
--host 0.0.0.0 --port 7088 --model meta-llama/Llama-3.2-1B-Instruct
```

```bash
curl -X 'POST' \
  'http://localhost:7088/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "model",
  "temperature": 0.9,
  "top_p": 0,
  "top_k": 0,
  "max_tokens": 256,
  "repetition_penalty": 1,
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "stream": true
}'
```

### Torch compile

```bash
python3 -m ecahlang \
--host 0.0.0.0 --port 7088 --model meta-llama/Llama-3.2-1B-Instruct \
--torch_compile true
```

#### Reduce overhead

```bash
python3 -m ecahlang \
--host 0.0.0.0 --port 7088 --model meta-llama/Llama-3.2-1B-Instruct \
--torch_compile true --torch_compile_mode reduce-overhead --max_sequence 10
```

#### Max autotune

```bash
python3 -m ecahlang \
--host 0.0.0.0 --port 7088 --model meta-llama/Llama-3.2-1B-Instruct \
--torch_compile true --torch_compile_mode max-autotune --max_sequence 10
```

## Architecture

```
ecahlang/
├── server_args.py                 # CLI args + env var config
├── entrypoints/                   # API layer (vLLM pattern)
│   ├── api_server.py              # FastAPI app, startup/shutdown
│   ├── protocol.py                # Pydantic request/response models
│   └── streaming.py               # SSE streaming, per-request coroutine
├── core/                          # Scheduler (vLLM v1/core pattern)
│   ├── scheduler.py               # Queue management, batch formation
│   └── request_state.py           # Per-request state tracking
├── mem/                           # KV cache management (SGLang mem_cache pattern)
│   ├── paged_kv_manager.py        # Paged KV cache with free-list
│   └── radix_kv_manager.py        # Radix tree with prefix sharing
├── model_executor/                # GPU execution (vLLM + SGLang pattern)
│   ├── model_runner.py            # Load HF model, forward pass
│   ├── attention.py               # FlashInfer attention hook
│   └── cuda_graph_runner.py       # CUDA graph capture/replay
├── sampling/                      # Sampling (SGLang pattern)
│   └── sampler.py                 # top-k, top-p, temperature, repetition penalty
└── managers/                      # Background managers (SGLang pattern)
    ├── detokenizer.py             # Background batch_decode in thread pool
    └── overlap.py                 # CUDA stream overlap
```
