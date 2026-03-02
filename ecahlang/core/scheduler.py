"""
Core scheduler: queue management, batch formation, and the process_queue loop.

This is the heart of continuous batching. Two background tasks run forever:
  - prefill loop: drains prefill_queue → batch prefill → resolve futures
  - decode loop:  drains step_queue → batch decode → resolve futures

Inspired by:
- vLLM v1/core/sched/ (scheduler)
- SGLang managers/scheduler.py
"""

import asyncio
import logging
import torch
import flashinfer
from contextlib import nullcontext

from ..model_executor.attention import set_attention_state
from ..sampling.sampler import logits_to_probs
from ..managers.detokenizer import background_batch_decode
from ..managers.overlap import OverlapManager

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Manages prefill and decode queues, forms batches, and runs the process loop.
    """

    def __init__(self, args, model_runner, kv_manager):
        self.args = args
        self.model_runner = model_runner
        self.kv_manager = kv_manager

        # Queues: each item is (future, input_ids, position, uuid, temperature, top_k, top_p, length)
        self.prefill_queue = asyncio.Queue()
        self.step_queue = asyncio.Queue()

        # FlashInfer workspace buffers
        workspace_prefill = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        workspace_decode = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_prefill, "NHD")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_decode, "NHD")

        # Constant tensors
        self.empty_length = torch.tensor([0]).cuda()
        self.decode_length = torch.tensor([1]).cuda()

        # Profiler
        if args.torch_profiling:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
        else:
            self.profiler = nullcontext()

        # CUDA stream overlap
        self.overlap = OverlapManager(enabled=True)

        # CUDA graph wrapper (set during startup warmup if enabled)
        self.cuda_graph_wrapper = None

        # Background tasks (set during startup)
        self._prefill_task = None
        self._step_task = None

    def start(self):
        """Start the prefill and decode background loops."""
        self._prefill_task = asyncio.create_task(self._prefill_loop())
        self._step_task = asyncio.create_task(self._step_loop())
        logger.info('Scheduler started: prefill and decode loops running')

    async def _prefill_loop(self):
        """Background loop that processes the prefill queue."""
        await self._process_queue(self.prefill_queue, self.prefill_wrapper, prefill=True)

    async def _step_loop(self):
        """Background loop that processes the decode (step) queue."""
        await self._process_queue(self.step_queue, self.decode_wrapper, prefill=False)

    async def _process_queue(self, queue, wrapper, prefill):
        """
        Core batch processing loop.

        Continuously drains the queue, forms batches, runs forward passes,
        samples tokens, and resolves futures.
        """
        global_step = 0
        need_sleep = True

        while True:
            if need_sleep:
                await asyncio.sleep(self.args.microsleep)

            need_sleep = True
            batch = []

            # Drain queue up to max_sequence
            while not queue.empty():
                try:
                    request = await asyncio.wait_for(queue.get(), timeout=1e-6)
                    batch.append(request)
                    if len(batch) >= self.args.max_sequence:
                        need_sleep = False
                        break
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            with self.profiler as prof:
                # Unpack batch
                futures, inputs, position_ids, uuids = zip(
                    *[(b[0], b[1], b[2], b[3]) for b in batch]
                )
                temperature, top_k, top_p, lengths = zip(
                    *[(b[4], b[5], b[6], b[7]) for b in batch]
                )
                lengths_cpu = [inp.shape[0] for inp in inputs]

                try:
                    # Build position IDs
                    position_ids = (
                        torch.cat([torch.arange(l) for l in lengths_cpu])
                        if prefill
                        else torch.tensor(position_ids)
                    )[None].cuda()

                    # Allocate/extend KV cache pages
                    for no, l in enumerate(lengths_cpu):
                        if prefill:
                            self.kv_manager.allocate(uuids[no], l)
                        else:
                            self.kv_manager.append_tokens(uuids[no], l)

                    # Concat all input_ids into single flat tensor
                    input_ids = torch.concat(inputs)[None]
                    lengths_tensor = torch.concat([self.empty_length] + list(lengths))
                    append_indptr = torch.cumsum(lengths_tensor, dim=-1).to(torch.int32)

                    # Get FlashInfer metadata
                    kv_indices, kv_indptr, kv_last_page_len = self.kv_manager.get_append_metadata(uuids)

                    # Plan FlashInfer attention pattern
                    if prefill:
                        wrapper.plan(
                            append_indptr,
                            kv_indptr,
                            kv_indices,
                            kv_last_page_len,
                            self.model_runner.num_heads,
                            self.model_runner.num_kv_heads,
                            self.model_runner.head_dim,
                            self.kv_manager.block_size,
                            causal=True,
                            q_data_type=self.args.torch_dtype,
                        )
                    else:
                        wrapper.plan(
                            kv_indptr,
                            kv_indices,
                            kv_last_page_len,
                            self.model_runner.num_heads,
                            self.model_runner.num_kv_heads,
                            self.model_runner.head_dim,
                            self.kv_manager.block_size,
                            pos_encoding_mode="NONE",
                            q_data_type=self.args.torch_dtype,
                        )

                    # Set attention state for the hook
                    set_attention_state(wrapper, self.kv_manager, prefill, append_indptr, self.args)

                    # Reset layer counters
                    if prefill:
                        self.kv_manager.prefill_layer_idx = 0
                        self.kv_manager.prefill_batch_ids = uuids
                    else:
                        self.kv_manager.decode_layer_idx = 0
                        self.kv_manager.decode_batch_ids = uuids

                    # Forward pass on compute stream (overlap GPU with CPU prep)
                    forward_fn = self.model_runner.forward if prefill else self.model_runner.decode_forward
                    with self.overlap.compute_stream():
                        output = forward_fn(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            wrapper=wrapper,
                            manager=self.kv_manager,
                            prefill=prefill,
                            append_indptr=append_indptr,
                        )

                    # Prepare sampling params on CPU while GPU may still be computing
                    temperature = torch.concat(temperature)[None].T
                    top_k = torch.concat(top_k)
                    top_p = torch.concat(top_p)

                    mask_penalty = []
                    for uuid in uuids:
                        mask_penalty.append(
                            self.kv_manager.mask_penalty[self.kv_manager.batch_to_seq_len[uuid]]
                        )
                    mask_penalty = torch.stack(mask_penalty)

                    # Sync: wait for forward pass to finish before reading logits
                    self.overlap.synchronize()

                    # Extract last-token logits per sequence
                    logits = output.logits[0, append_indptr[1:] - 1]

                    # Sample next tokens
                    idx_next = logits_to_probs(logits, mask_penalty, temperature, top_k, top_p)

                    # Decode tokens to text (in background thread)
                    tokens = await background_batch_decode(
                        self.model_runner.tokenizer, idx_next,
                    )

                    # Resolve futures
                    for i, fut in enumerate(futures):
                        fut.set_result((idx_next[i], tokens[i]))

                except Exception as e:
                    logger.error(f'Error in process_queue: {e}', exc_info=True)
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

            # Export profiling trace
            if self.args.torch_profiling:
                try:
                    mode = 'prefill' if prefill else 'decode'
                    prof.export_chrome_trace(f'{mode}-{global_step}.json')
                except Exception as e:
                    logger.warning(f'Failed to export trace: {e}')

            global_step += 1

    async def enqueue_prefill(self, future, input_ids, position, uuid, temperature, top_k, top_p, length):
        """Put a prefill request into the prefill queue."""
        await self.prefill_queue.put(
            (future, input_ids, position, uuid, temperature, top_k, top_p, length)
        )

    async def enqueue_step(self, future, input_ids, position, uuid, temperature, top_k, top_p, length):
        """Put a decode step request into the step queue."""
        await self.step_queue.put(
            (future, input_ids, position, uuid, temperature, top_k, top_p, length)
        )
