"""
Per-request state tracking.

Each incoming request creates a RequestState that tracks:
- Token history and generation progress
- Sampling parameters (temperature, top_k, top_p, repetition_penalty)
- Asyncio futures for coroutine suspension/resumption
- Timing info

Inspired by:
- vLLM sequence.py (SequenceData, Sequence)
- SGLang managers/schedule_batch.py (Req)
"""

import time
import torch


class RequestState:
    """
    Tracks the lifecycle of a single inference request.

    Created when a request arrives, destroyed when it completes (EOS or max_tokens).
    """

    def __init__(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
        ignore_eos: bool,
        vocab_size: int,
    ):
        self.request_id = request_id
        self.input_ids = input_ids.cuda()
        self.initial_length = input_ids.shape[0]
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

        # Sampling params as CUDA tensors (ready for batched sampling)
        self.temperature = torch.tensor([max(1e-5, temperature)]).cuda()
        self.top_k = torch.tensor([vocab_size if top_k == 0 else top_k], dtype=torch.int32).cuda()
        self.top_p = torch.tensor([1.0 if top_p == 0 else top_p], dtype=torch.float32).cuda()
        self.repetition_penalty = max(1e-5, repetition_penalty)

        # Generation state
        self.generated_tokens = 0
        self.current_input = input_ids.cuda()  # starts with full input, then single tokens

        # Timing
        self.start_time = time.perf_counter()
        self.time_first_token = None

    @property
    def current_position(self):
        """Current position in the sequence (input_length + generated_tokens)."""
        return self.initial_length + self.generated_tokens

    @property
    def current_length(self):
        """Length of current input to feed to the model."""
        return self.current_input.shape[0]

    @property
    def is_prefill(self):
        """True if this is the first iteration (prefill), False for decode."""
        return self.generated_tokens == 0

    def advance(self, next_token_id: torch.Tensor):
        """Update state after generating a token."""
        self.generated_tokens += 1
        self.current_input = next_token_id
        if self.generated_tokens == 1:
            self.time_first_token = time.perf_counter()

    @property
    def total_tokens(self):
        """Total tokens processed (input + generated)."""
        return self.initial_length + self.generated_tokens

    @property
    def duration(self):
        """Time elapsed since request start."""
        return time.perf_counter() - self.start_time

    @property
    def tokens_per_second(self):
        """Generation speed."""
        d = self.duration
        if d == 0:
            return 0
        return self.total_tokens / d
