"""
CUDA stream overlap manager for compute/communication.

Allows GPU compute (model forward) to overlap with CPU work
(sampling, queue management, tokenizer decode) using separate CUDA streams.

Without overlap:
  [GPU: forward] → [CPU: sample + decode + queue] → [GPU: forward]
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    GPU idle during CPU work

With overlap:
  Stream 1: [GPU: forward] ──────────────── [GPU: forward]
  Stream 0:                 [CPU: sample]
                            overlapped

Inspired by SGLang managers/overlap_utils.py
"""

import torch


class OverlapManager:
    """
    Manages CUDA streams for overlapping compute and communication.

    Usage:
        overlap = OverlapManager()
        with overlap.compute_stream():
            output = model(input_ids, position_ids)
        overlap.synchronize()
        # Now safe to read output on default stream
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        if enabled and torch.cuda.is_available():
            self._compute_stream = torch.cuda.Stream()
        else:
            self._compute_stream = None

    def compute_stream(self):
        """Context manager for running GPU compute on a separate stream."""
        if self._compute_stream is not None:
            return torch.cuda.stream(self._compute_stream)
        return _NullContext()

    def synchronize(self):
        """Wait for compute stream to finish before reading results."""
        if self._compute_stream is not None:
            self._compute_stream.synchronize()

    def record_event(self):
        """Record an event on the compute stream for fine-grained sync."""
        if self._compute_stream is not None:
            event = torch.cuda.Event()
            event.record(self._compute_stream)
            return event
        return None

    def wait_event(self, event):
        """Wait for an event on the default stream."""
        if event is not None:
            torch.cuda.current_stream().wait_event(event)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
