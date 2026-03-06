import torch
import math
import pynvml
import flashinfer
import os

def get_total_free_memory(index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free

class AutoKVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int = 10000,
        seq_lens: int = 128,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        layout: str = "NHD",
        total_gpu_mem_bytes: int = None,
        mem_utilization: float = 0.9,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.seq_lens = max(seq_lens, 128)
        self.block_size = block_size
        self.dtype = dtype
        self.layout = layout.upper()
        self.dtype_size = torch.tensor([], dtype=dtype).element_size()
        self.mask_penalty = torch.ones(self.seq_lens, vocab_size, dtype=dtype).cuda()

        if total_gpu_mem_bytes is None:
            devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if devices is None:
                devices = list(range(torch.cuda.device_count()))
            else:
                devices = [d.strip() for d in devices.split(',')]
            total_gpu_mem_bytes = get_total_free_memory(int(devices[0]))
        
        usable_mem = int(total_gpu_mem_bytes * mem_utilization)
        per_token_bytes = num_kv_heads * head_dim * self.dtype_size * 2
        per_block_per_layer = block_size * per_token_bytes
        self.max_blocks = usable_mem // (num_layers * per_block_per_layer)

        self.kv_cache = torch.zeros(
            (num_layers, self.max_blocks, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype, device="cuda"
        )

        self.free_blocks = list(range(self.max_blocks))
        self.free_seq_lens = list(range(self.seq_lens))
        self.batch_to_blocks = {}
        self.batch_to_page_lengths = {}
        self.batch_to_total_tokens = {}
        self.batch_to_seq_len = {}
        self.prefill_layer_idx = 0
        self.decode_layer_idx = 0

        self.cuda_graph_mode = False
        self.padding_pages = []
        self._cg_kv_indices = {}
        self._cg_kv_indptr = {}
        self._cg_kv_last_page_len = {}
        self._cg_batch_indices = {}
        self._cg_positions = {}
        self._cg_append_indptr = {}

    def init_sampling_buffers(self, max_batch_size):
        # Pinned CPU buffers (for fast async H2D copy)
        self.sampling_temperature_cpu = torch.ones(max_batch_size, 1, dtype=torch.float32).pin_memory()
        self.sampling_top_k_cpu = torch.full((max_batch_size,), self.vocab_size, dtype=torch.int32).pin_memory()
        self.sampling_top_p_cpu = torch.ones(max_batch_size, dtype=torch.float32).pin_memory()

        # Numpy views for zero-overhead scalar writes
        self.sampling_temperature_np = self.sampling_temperature_cpu.numpy()
        self.sampling_top_k_np = self.sampling_top_k_cpu.numpy()
        self.sampling_top_p_np = self.sampling_top_p_cpu.numpy()

        # GPU buffers (persistent, reused every step)
        self.sampling_temperature_gpu = torch.ones(max_batch_size, 1, dtype=torch.float32, device="cuda")
        self.sampling_top_k_gpu = torch.full((max_batch_size,), self.vocab_size, dtype=torch.int32, device="cuda")
        self.sampling_top_p_gpu = torch.ones(max_batch_size, dtype=torch.float32, device="cuda")

    def fill_sampling_params(self, n, temperatures, top_ks, top_ps):
        self.sampling_temperature_np[:n, 0] = temperatures
        self.sampling_top_k_np[:n] = top_ks
        self.sampling_top_p_np[:n] = top_ps

        self.sampling_temperature_gpu[:n].copy_(self.sampling_temperature_cpu[:n], non_blocking=True)
        self.sampling_top_k_gpu[:n].copy_(self.sampling_top_k_cpu[:n], non_blocking=True)
        self.sampling_top_p_gpu[:n].copy_(self.sampling_top_p_cpu[:n], non_blocking=True)

    def allocate(self, batch_id, total_tokens):
        num_pages = math.ceil(total_tokens / self.block_size)
        if len(self.free_blocks) < num_pages:
            raise RuntimeError("Not enough KV cache blocks available")

        blocks = [self.free_blocks.pop() for _ in range(num_pages)]
        self.batch_to_blocks[batch_id] = blocks
        self.batch_to_page_lengths[batch_id] = total_tokens % self.block_size
        self.batch_to_total_tokens[batch_id] = total_tokens

        seq_len = self.free_seq_lens.pop()
        self.batch_to_seq_len[batch_id] = seq_len
        self.mask_penalty[seq_len] = 1.0
        return blocks

    def free(self, batch_id):
        blocks = self.batch_to_blocks.pop(batch_id, [])
        self.free_blocks.extend(blocks)
        self.batch_to_page_lengths.pop(batch_id, None)
        seq_len = self.batch_to_seq_len.pop(batch_id, None)
        if seq_len is not None:
            self.free_seq_lens.append(seq_len)

    def get_append_metadata(self, batch_ids):
        """Returns kv_indices, kv_indptr, kv_last_page_len for FlashInfer append."""
        kv_indices = []
        kv_indptr = [0]
        kv_last_page_len = []

        for bid in batch_ids:
            pages = self.batch_to_blocks[bid]
            kv_indices.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            kv_last_page_len.append(self.batch_to_page_lengths[bid])

        return (
            torch.tensor(kv_indices, dtype=torch.int32, device="cuda"),
            torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"),
            torch.tensor(kv_last_page_len, dtype=torch.int32, device="cuda"),
        )
    
    def prepare_append_metadata(self, batch_ids, append_indptr):
        """Pre-compute append metadata once per batch, reused across all layers."""
        kv_indices, kv_indptr, kv_last_page_len = self.get_append_metadata(batch_ids)
        seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, self.block_size)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens, append_indptr[-1]
        )
        self._cached_kv_indices = kv_indices
        self._cached_kv_indptr = kv_indptr
        self._cached_kv_last_page_len = kv_last_page_len
        self._cached_batch_indices = batch_indices
        self._cached_positions = positions

    def append_paged_kv_cache_cached(self, key, value, layer_idx):
        """Append using pre-computed metadata (call prepare_append_metadata first)."""
        flashinfer.page.append_paged_kv_cache(
            append_key=key,
            append_value=value,
            batch_indices=self._cached_batch_indices,
            positions=self._cached_positions,
            paged_kv_cache=self.kv_cache[layer_idx],
            kv_indices=self._cached_kv_indices,
            kv_indptr=self._cached_kv_indptr,
            kv_last_page_len=self._cached_kv_last_page_len,
            kv_layout=self.layout,
        )

    def append_paged_kv_cache(self, batch_ids, key, value, append_indptr, layer_idx):
        kv_indices, kv_indptr, kv_last_page_len = self.get_append_metadata(batch_ids)

        seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, self.block_size)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens, append_indptr[-1]
        )

        flashinfer.page.append_paged_kv_cache(
            append_key=key,
            append_value=value,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=self.kv_cache[layer_idx],
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout=self.layout,
        )

    def append_tokens(self, batch_id, num_new_tokens):

        if batch_id not in self.batch_to_blocks:
            raise ValueError(f"{batch_id} not allocated")

        current_len = self.batch_to_page_lengths.get(batch_id, 0)
        total_tokens = current_len + num_new_tokens

        full_new_pages = total_tokens // self.block_size
        remaining = total_tokens % self.block_size

        if full_new_pages > len(self.free_blocks):
            raise RuntimeError("Not enough free blocks to append tokens")

        for _ in range(full_new_pages):
            new_page = self.free_blocks.pop()
            self.batch_to_blocks[batch_id].append(new_page)

        self.batch_to_page_lengths[batch_id] = remaining

    def init_cuda_graph_buffers(self, bucket_sizes):
        max_bucket = max(bucket_sizes)

        if len(self.free_blocks) < max_bucket:
            raise RuntimeError(f"Not enough free blocks for {max_bucket} padding pages")
        self.padding_pages = [self.free_blocks.pop() for _ in range(max_bucket)]

        self._cg_temperature = {}
        self._cg_top_k = {}
        self._cg_top_p = {}
        for bs in bucket_sizes:
            self._cg_kv_indices[bs] = torch.zeros(self.max_blocks, dtype=torch.int32, device="cuda")
            self._cg_kv_indptr[bs] = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
            self._cg_kv_last_page_len[bs] = torch.zeros(bs, dtype=torch.int32, device="cuda")
            self._cg_batch_indices[bs] = torch.arange(bs, dtype=torch.int32, device="cuda")
            self._cg_positions[bs] = torch.zeros(bs, dtype=torch.int32, device="cuda")
            self._cg_append_indptr[bs] = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
            self._cg_temperature[bs] = torch.ones(bs, 1, dtype=torch.float32, device="cuda")
            self._cg_top_k[bs] = torch.full((bs,), self.vocab_size, dtype=torch.int32, device="cuda")
            self._cg_top_p[bs] = torch.ones(bs, dtype=torch.float32, device="cuda")

    def fill_cuda_graph_metadata(self, bucket_size, real_uuids):
        num_real = len(real_uuids)

        kv_indices, kv_indptr, kv_last_page_len = [], [0], []
        positions = []

        for uid in real_uuids:
            pages = self.batch_to_blocks[uid]
            kv_indices.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            kv_last_page_len.append(self.batch_to_page_lengths[uid])
            total_tokens = (len(pages) - 1) * self.block_size + self.batch_to_page_lengths[uid]
            positions.append(total_tokens - 1)

        for i in range(bucket_size - num_real):
            kv_indices.append(self.padding_pages[i])
            kv_indptr.append(kv_indptr[-1] + 1)
            kv_last_page_len.append(1)
            positions.append(0)

        kv_idx_t = torch.tensor(kv_indices, dtype=torch.int32, device="cuda")
        n_idx = len(kv_indices)
        self._cg_kv_indices[bucket_size][:n_idx].copy_(kv_idx_t)
        self._cg_kv_indptr[bucket_size].copy_(torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"))
        self._cg_kv_last_page_len[bucket_size].copy_(torch.tensor(kv_last_page_len, dtype=torch.int32, device="cuda"))
        self._cg_positions[bucket_size].copy_(torch.tensor(positions, dtype=torch.int32, device="cuda"))

    def fill_cuda_graph_sampling_params(self, bucket_size, n, temperatures, top_ks, top_ps):
        # Reset to safe defaults (padding won't affect real results)
        self._cg_temperature[bucket_size].fill_(1.0)
        self._cg_top_k[bucket_size].fill_(self.vocab_size)
        self._cg_top_p[bucket_size].fill_(1.0)

        # Write real values via pinned CPU
        self.sampling_temperature_np[:n, 0] = temperatures
        self.sampling_top_k_np[:n] = top_ks
        self.sampling_top_p_np[:n] = top_ps

        self._cg_temperature[bucket_size][:n].copy_(self.sampling_temperature_cpu[:n], non_blocking=True)
        self._cg_top_k[bucket_size][:n].copy_(self.sampling_top_k_cpu[:n], non_blocking=True)
        self._cg_top_p[bucket_size][:n].copy_(self.sampling_top_p_cpu[:n], non_blocking=True)

    def append_paged_kv_cache_cuda_graph(self, key, value, bucket_size, layer_idx):
        flashinfer.page.append_paged_kv_cache(
            append_key=key,
            append_value=value,
            batch_indices=self._cg_batch_indices[bucket_size],
            positions=self._cg_positions[bucket_size],
            paged_kv_cache=self.kv_cache[layer_idx],
            kv_indices=self._cg_kv_indices[bucket_size],
            kv_indptr=self._cg_kv_indptr[bucket_size],
            kv_last_page_len=self._cg_kv_last_page_len[bucket_size],
            kv_layout=self.layout,
        )