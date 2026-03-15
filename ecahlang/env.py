import argparse
import logging
import os
import torch

torch.set_grad_enabled(False)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--max_sequence', type=int,
        default=int(os.environ.get('MAX_SEQUENCE', '128')),
        help='max sequence aka batch size per filling or decoding (default: %(default)s, env: MAX_SEQUENCE)'
    )
    parser.add_argument(
        '--memory_utilization', type=float,
        default=float(os.environ.get('MEMORY_UTILIZATION', '0.9')),
        help='memory utilization on free memory after load the model for automatic number of paging for paged attention (default: %(default)s, env: MEMORY_UTILIZATION)'
    )
    parser.add_argument(
        '--compare-sdpa-prefill', type=lambda x: x.lower() == 'true',
        default=os.environ.get('COMPARE_SDPA_PREFILL', 'false').lower() == 'true',
        help='Compare FlashInfer attention output with SDPA during prefill (default: %(default)s, env: COMPARE_SDPA_PREFILL)'
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        help='Model type (default: %(default)s, env: MODEL)'
    )
    parser.add_argument(
        '--torch_dtype',
        default=os.environ.get('TORCH_DTYPE', 'float16'),
        help='Model dtype (default: %(default)s, env: TORCH_DTYPE)'
    )
    parser.add_argument(
        '--torch_dtype_autocast',
        default=os.environ.get('TORCH_DTYPE_AUTOCAST', 'float16'),
        help='Model dtype autocast if the model loaded in float32 (default: %(default)s, env: TORCH_DTYPE_AUTOCAST)'
    )
    parser.add_argument(
        '--torch_profiling',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_PROFILING', 'false').lower() == 'true',
        help='Use torch.autograd.profiler.profile() to profile prefill and step (default: %(default)s, env: TORCH_PROFILING)'
    )
    parser.add_argument(
        '--torch_compile',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_COMPILE', 'false').lower() == 'true',
        help='Torch compile for decoding and sampling (default: %(default)s, env: TORCH_COMPILE)'
    )
    parser.add_argument(
        '--torch_compile_mode',
        default=os.environ.get('TORCH_COMPILE_MODE', 'default'),
        help='torch compile mode (default: %(default)s, env: TORCH_COMPILE_MODE)'
    )
    parser.add_argument(
        '--cuda_graph',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('CUDA_GRAPH', 'false').lower() == 'true',
        help='Capture CUDA Graph for decoding and sampling (default: %(default)s, env: CUDA_GRAPH)'
    )
    parser.add_argument(
        '--multi_step', default=int(os.environ.get('MULTI_STEP', '1')), type=int,
        help='Number of decode steps to run per round-trip (default: %(default)s, env: MULTI_STEP)'
    )
    parser.add_argument(
        '--max_prefill_tokens', type=int,
        default=int(os.environ.get('MAX_PREFILL_TOKENS', '2048')),
        help='Max total tokens per prefill batch for chunked prefill, 0 to disable (default: %(default)s, env: MAX_PREFILL_TOKENS)'
    )
    parser.add_argument(
        '--skip_batch_decode',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('SKIP_BATCH_DECODE', 'false').lower() == 'true',
        help='Call tokenizer.batch_decode synchronously instead of via run_in_executor, useful for throughput benchmarking (default: %(default)s, env: SKIP_BATCH_DECODE)'
    )
    parser.add_argument(
        '--block_size', type=int,
        default=int(os.environ.get('BLOCK_SIZE', '16')),
        help='KV cache page block size in tokens (default: %(default)s, env: BLOCK_SIZE)'
    )
    parser.add_argument(
        '--overlap_logging',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('OVERLAP_LOGGING', 'false').lower() == 'true',
        help='Enable communication/computation overlap timing logs into timing_logs global (introduces latency, disabled by default) (default: %(default)s, env: OVERLAP_LOGGING)'
    )

    args = parser.parse_args()

    if args.torch_compile and args.cuda_graph:
        raise ValueError('cannot set both torch compile and CUDA Graph')

    if args.cuda_graph and args.multi_step > 1:
        raise ValueError('CUDA graph does not support multi_step > 1')

    if args.torch_dtype not in {'float16', 'bfloat16', 'float32'}:
        raise ValueError('`--torch_dtype` only support `float16` or `bfloat16` or `float32`')

    if args.torch_dtype == 'float32':
        if args.torch_dtype_autocast not in {'float16', 'bfloat16'}:
            raise ValueError('`--torch_dtype_autocast` only support `float16` or `bfloat16` or `float32`')

        args.model_dtype = getattr(torch, args.torch_dtype)
        args.torch_dtype = getattr(torch, args.torch_dtype_autocast)
    else:
        args.torch_dtype = getattr(torch, args.torch_dtype)
        args.model_dtype = args.torch_dtype

    args.device = 'cuda'
    args.need_autocast = args.model_dtype == torch.float32

    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

if args.need_autocast:
    logging.info(f'Model loaded in float32, during attention forward it will autocast to {args.torch_dtype}')

if args.torch_compile:
    logging.warning('torch compile is not optimize for big batch')

logging.info(f'Serving app using {args}')
