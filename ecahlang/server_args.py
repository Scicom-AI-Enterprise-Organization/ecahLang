import argparse
import logging
import os
import torch

torch.set_grad_enabled(False)


def parse_arguments():
    parser = argparse.ArgumentParser(description='ecahLang - Continuous Batching LLM Inference')

    parser.add_argument(
        '--host', type=str,
        default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)',
    )
    parser.add_argument(
        '--port', type=int,
        default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)',
    )
    parser.add_argument(
        '--loglevel',
        default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)',
    )
    parser.add_argument(
        '--microsleep', type=float,
        default=float(os.environ.get('MICROSLEEP', '1e-4')),
        help='microsleep to group batching, 1 / 1e-4 = 10k steps/sec (default: %(default)s, env: MICROSLEEP)',
    )
    parser.add_argument(
        '--max_sequence', type=int,
        default=int(os.environ.get('MAX_SEQUENCE', '128')),
        help='max batch size per prefill or decode step (default: %(default)s, env: MAX_SEQUENCE)',
    )
    parser.add_argument(
        '--memory_utilization', type=float,
        default=float(os.environ.get('MEMORY_UTILIZATION', '0.9')),
        help='fraction of free GPU memory for KV cache pages (default: %(default)s, env: MEMORY_UTILIZATION)',
    )
    parser.add_argument(
        '--compare-sdpa-prefill',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('COMPARE_SDPA_PREFILL', 'false').lower() == 'true',
        help='compare FlashInfer attention output with SDPA during prefill (default: %(default)s, env: COMPARE_SDPA_PREFILL)',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        help='HuggingFace model name or path (default: %(default)s, env: MODEL)',
    )
    parser.add_argument(
        '--torch_dtype',
        default=os.environ.get('TORCH_DTYPE', 'float16'),
        help='model dtype: float16, bfloat16, float32 (default: %(default)s, env: TORCH_DTYPE)',
    )
    parser.add_argument(
        '--torch_dtype_autocast',
        default=os.environ.get('TORCH_DTYPE_AUTOCAST', 'float16'),
        help='autocast dtype when model is float32 (default: %(default)s, env: TORCH_DTYPE_AUTOCAST)',
    )
    parser.add_argument(
        '--torch_profiling',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_PROFILING', 'false').lower() == 'true',
        help='profile prefill and step with torch profiler (default: %(default)s, env: TORCH_PROFILING)',
    )
    parser.add_argument(
        '--torch_compile',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('TORCH_COMPILE', 'false').lower() == 'true',
        help='torch.compile for decode (default: %(default)s, env: TORCH_COMPILE)',
    )
    parser.add_argument(
        '--torch_compile_mode',
        default=os.environ.get('TORCH_COMPILE_MODE', 'default'),
        help='torch.compile mode (default: %(default)s, env: TORCH_COMPILE_MODE)',
    )
    parser.add_argument(
        '--cuda_graph',
        type=lambda x: x.lower() == 'true',
        default=os.environ.get('CUDA_GRAPH', 'false').lower() == 'true',
        help='capture CUDA Graph for decode (default: %(default)s, env: CUDA_GRAPH)',
    )

    args = parser.parse_args()

    if args.torch_compile and args.cuda_graph:
        raise ValueError('cannot set both torch_compile and cuda_graph')

    if args.torch_dtype not in {'float16', 'bfloat16', 'float32'}:
        raise ValueError('--torch_dtype only supports float16, bfloat16, float32')

    if args.torch_dtype == 'float32':
        if args.torch_dtype_autocast not in {'float16', 'bfloat16'}:
            raise ValueError('--torch_dtype_autocast only supports float16, bfloat16')
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
    logging.info(f'Model loaded in float32, attention forward will autocast to {args.torch_dtype}')

if args.torch_compile:
    logging.warning('torch compile is not optimized for large batches')

logging.info(f'ecahLang serving with {args}')
