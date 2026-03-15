# ecahLang

Lightweight continuous batching inference engine for HuggingFace CausalLM models, built on top of [FlashInfer](https://github.com/flashinfer-ai/flashinfer). The name "ecah" is inspired by Aisyah, a lovely and popular Malaysian girl's name, a short and friendly nickname.

## Features

- Continuous batching with paged KV cache
- FlashInfer paged prefill and decode attention
- CUDA Graph decode
- torch.compile decode
- CUDA stream overlap schedule
- Pre-allocated pinned sampling buffers
- Chunked prefill

## Pre-requisites

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate
```

## Installation

```bash
pip install git+https://github.com/Scicom-AI-Enterprise-Organization/ecahLang
```

Or from source:

```bash
git clone https://github.com/Scicom-AI-Enterprise-Organization/ecahLang && cd ecahLang
pip install -e .
```

## Quick Start

Run ecahLang with CUDA Graph enabled for the best decode performance. The server will warm up FlashInfer and capture decode graphs at startup, then serve requests on an OpenAI-compatible API.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m ecahlang \
  --model Qwen/Qwen2.5-3B-Instruct \
  --torch_dtype float16 \
  --host 0.0.0.0 --port 7088 \
  --memory_utilization 0.5 \
  --cuda_graph true
```

```bash
curl -X POST http://localhost:7088/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

## Supported Parameters

```bash
python3 -m ecahlang --help
```

```text
usage: __main__.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--max_sequence MAX_SEQUENCE] [--memory_utilization MEMORY_UTILIZATION]
                   [--compare-sdpa-prefill COMPARE_SDPA_PREFILL] [--model MODEL] [--torch_dtype TORCH_DTYPE] [--torch_dtype_autocast TORCH_DTYPE_AUTOCAST]
                   [--torch_profiling TORCH_PROFILING] [--torch_compile TORCH_COMPILE] [--torch_compile_mode TORCH_COMPILE_MODE] [--cuda_graph CUDA_GRAPH]
                   [--multi_step MULTI_STEP] [--max_prefill_tokens MAX_PREFILL_TOKENS] [--skip_batch_decode SKIP_BATCH_DECODE] [--block_size BLOCK_SIZE]
                   [--overlap_logging OVERLAP_LOGGING]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --max_sequence MAX_SEQUENCE
                        max sequence aka batch size per filling or decoding (default: 128, env: MAX_SEQUENCE)
  --memory_utilization MEMORY_UTILIZATION
                        memory utilization on free memory after load the model for automatic number of paging for paged attention (default: 0.9, env:
                        MEMORY_UTILIZATION)
  --compare-sdpa-prefill COMPARE_SDPA_PREFILL
                        Compare FlashInfer attention output with SDPA during prefill (default: False, env: COMPARE_SDPA_PREFILL)
  --model MODEL         Model type (default: meta-llama/Llama-3.2-1B-Instruct, env: MODEL)
  --torch_dtype TORCH_DTYPE
                        Model dtype (default: float16, env: TORCH_DTYPE)
  --torch_dtype_autocast TORCH_DTYPE_AUTOCAST
                        Model dtype autocast if the model loaded in float32 (default: float16, env: TORCH_DTYPE_AUTOCAST)
  --torch_profiling TORCH_PROFILING
                        Use torch.autograd.profiler.profile() to profile prefill and step (default: False, env: TORCH_PROFILING)
  --torch_compile TORCH_COMPILE
                        Torch compile for decoding and sampling (default: False, env: TORCH_COMPILE)
  --torch_compile_mode TORCH_COMPILE_MODE
                        torch compile mode (default: default, env: TORCH_COMPILE_MODE)
  --cuda_graph CUDA_GRAPH
                        Capture CUDA Graph for decoding and sampling (default: False, env: CUDA_GRAPH)
  --multi_step MULTI_STEP
                        Number of decode steps to run per round-trip (default: 1, env: MULTI_STEP)
  --max_prefill_tokens MAX_PREFILL_TOKENS
                        Max total tokens per prefill batch for chunked prefill, 0 to disable (default: 2048, env: MAX_PREFILL_TOKENS)
  --skip_batch_decode SKIP_BATCH_DECODE
                        Call tokenizer.batch_decode synchronously instead of via run_in_executor, useful for throughput benchmarking (default: False, env:
                        SKIP_BATCH_DECODE)
  --block_size BLOCK_SIZE
                        KV cache page block size in tokens (default: 16, env: BLOCK_SIZE)
  --overlap_logging OVERLAP_LOGGING
                        Enable communication/computation overlap timing logs into timing_logs global (introduces latency, disabled by default) (default: False,
                        env: OVERLAP_LOGGING)
```

**We support both args and OS environment.**

## Benchmarks

Benchmarked on **Qwen/Qwen2.5-3B-Instruct** (float16) on a single **H100 SXM** across concurrency levels 1–128, compared against baseline (no optimizations), SGLang, and vLLM. Each request generates 384 tokens with `ignore_eos: true`. vLLM and SGLang were run with prefix caching disabled for a fair comparison.

### Time to First Token (TTFT)

![TTFT vs Concurrency](pics/ttft-graph-2.png)

### Inter-Token Latency (ITL)

![ITL vs Concurrency](pics/itl-graph-2.png)

### End-to-End Latency (E2E)

![E2E vs Concurrency](pics/e2e-graph-2.png)

### Running the Benchmark

```bash
# Against ecahLang
python3 benchmark/benchmark.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --save benchmark/my-results \
  --concurrency-list 1,2,4,8,16,32,64,128

# Against vLLM/SGLang (uses /v1/completions)
python3 benchmark/benchmark.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --save benchmark/vllm-results \
  --vllm \
  --concurrency-list 1,2,4,8,16,32,64,128
```

Visualization notebook: [`benchmark/benchmark_visualization.ipynb`](benchmark/benchmark_visualization.ipynb)

## Project Structure

```
ecahlang/
├── env.py          # CLI args + environment variable config
├── main.py         # FastAPI app, process_queue, stream, startup
├── manager.py      # Paged KV cache manager + sampling buffers
├── parameters.py   # Pydantic request/response models
└── utils.py        # Attention mask utilities
benchmark/
├── benchmark.py                 # Stress test script
├── benchmark_visualization.ipynb # Visualization notebook
└── <results>/                   # JSON results per concurrency level
pics/                            # Benchmark graphs
```
