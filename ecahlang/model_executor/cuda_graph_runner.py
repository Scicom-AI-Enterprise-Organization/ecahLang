"""
CUDA Graph capture and replay for decode path.

CUDA graphs eliminate CPU→GPU kernel launch overhead by recording a fixed
sequence of GPU operations and replaying them with minimal CPU involvement.

Inspired by:
- vLLM v1/cudagraph_dispatcher.py
- SGLang model_executor/cuda_graph_runner.py
"""

import logging
import torch

logger = logging.getLogger(__name__)


class CUDAGraphDecodeWrapper:
    """
    Captures and replays CUDA graphs for the decode forward pass.

    Usage:
        wrapper = CUDAGraphDecodeWrapper(model_runner.decode_forward)
        wrapper.warmup(batch_size=1, input_ids=..., position_ids=...)
        output = wrapper.run(batch_size=1, input_ids=new_ids, position_ids=new_pos)
    """

    def __init__(self, decode_fn):
        self.decode_fn = decode_fn
        self.graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}

    def warmup(self, key, **inputs):
        """Capture a CUDA graph for the given input shapes."""
        for k, v in inputs.items():
            inputs[k] = v.contiguous()

        self.static_inputs[key] = {k: v.clone().cuda() for k, v in inputs.items()}
        self.static_outputs[key] = {}

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.static_outputs[key] = self.decode_fn(**self.static_inputs[key])
        self.graphs[key] = g

        logger.debug(f'CUDA graph captured for key={key}')

    def run(self, key, **new_inputs):
        """Replay a captured CUDA graph with new inputs."""
        for k in new_inputs:
            if isinstance(new_inputs[k], torch.Tensor):
                self.static_inputs[key][k].copy_(new_inputs[k])
        self.graphs[key].replay()
        return self.static_outputs[key]

    def has_graph(self, key):
        """Check if a graph exists for the given key."""
        return key in self.graphs
