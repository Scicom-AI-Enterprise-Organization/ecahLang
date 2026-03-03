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