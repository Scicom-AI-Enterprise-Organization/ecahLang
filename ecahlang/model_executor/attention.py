"""
FlashInfer attention hook for HuggingFace Transformers.

Registered via AttentionInterface.register() to replace the model's
default attention with FlashInfer's paged attention kernels.

Data flow per layer:
  1. Receives query, key, value from HF attention layer [1, H, L, D]
  2. Reshapes to FlashInfer format [L, H, D]
  3. If FP32: downcasts to FP16/BF16 for FlashInfer kernels
  4. Appends K/V to paged cache (layer-by-layer tracking)
  5. Runs FlashInfer prefill or decode wrapper
  6. If was FP32: upcasts output back
  7. Returns [1, L, H, D]
"""

import torch
import logging
from transformers import AttentionInterface

from ..sampling.sampler import block_diagonal_concat_inverted

logger = logging.getLogger(__name__)

# Module-level state set by the scheduler before each forward pass
_attention_state = {
    'wrapper': None,
    'manager': None,
    'prefill': None,
    'append_indptr': None,
    'args': None,
}


def set_attention_state(wrapper, manager, prefill, append_indptr, args):
    """Called by scheduler before model forward to configure attention."""
    _attention_state['wrapper'] = wrapper
    _attention_state['manager'] = manager
    _attention_state['prefill'] = prefill
    _attention_state['append_indptr'] = append_indptr
    _attention_state['args'] = args


@torch.compiler.disable
def ecahlang_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    """
    Custom attention hook for FlashInfer paged attention.

    For prefilling: uses BatchPrefillWithPagedKVCacheWrapper
    For step decoding: uses BatchDecodeWithPagedKVCacheWrapper

    Input shape:  query/key/value = [1, H, L, D]
    Output shape: [1, L, H, D]
    """
    wrapper = kwargs.get('wrapper') or _attention_state['wrapper']
    manager = kwargs.get('manager') or _attention_state['manager']
    prefill = kwargs.get('prefill') if 'prefill' in kwargs else _attention_state['prefill']
    append_indptr = kwargs.get('append_indptr') or _attention_state['append_indptr']
    args = kwargs.get('args') or _attention_state['args']

    # Downcast FP32 → FP16/BF16 for FlashInfer kernels
    if args.need_autocast:
        query = query.to(args.torch_dtype)
        key = key.to(args.torch_dtype)
        value = value.to(args.torch_dtype)

    # Reshape: [1, H, L, D] → [L, H, D]
    query = query[0].transpose(0, 1)
    key = key[0].transpose(0, 1)
    value = value[0].transpose(0, 1)

    # Track which layer we're on
    layer_attr = 'prefill_layer_idx' if prefill else 'decode_layer_idx'
    layer_idx = getattr(manager, layer_attr)

    batch_attr = 'prefill_batch_ids' if prefill else 'decode_batch_ids'
    batch_ids = getattr(manager, batch_attr)

    # Write K/V into paged cache for this layer
    manager.append_paged_kv_cache(batch_ids, key, value, append_indptr, layer_idx)

    # Run FlashInfer attention kernel
    o = wrapper.run(query, manager.kv_cache[layer_idx])

    # Optional: compare with SDPA for debugging
    if args.compare_sdpa_prefill and prefill:
        _compare_with_sdpa(query, key, value, append_indptr, o, layer_idx)

    # Advance layer counter
    setattr(manager, layer_attr, layer_idx + 1)

    # Reshape: [L, H, D] → [1, L, H, D]
    o = o[None]

    # Upcast back to FP32 if needed
    if args.need_autocast:
        o = o.to(args.model_dtype)

    return o, None


def _compare_with_sdpa(query, key, value, append_indptr, flashinfer_output, layer_idx):
    """Debug helper: compare FlashInfer output with torch SDPA."""
    diff = torch.diff(append_indptr)
    masks = [torch.tril(torch.ones(l, l)) for l in diff]
    masks = block_diagonal_concat_inverted(*masks, dtype=query.dtype).cuda()

    q = query.transpose(0, 1)[None]
    k = key.transpose(0, 1)[None]
    v = value.transpose(0, 1)[None]
    enable_gqa = q.shape[1] != k.shape[1]

    output_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=enable_gqa,
    )
    output_sdpa = output_sdpa[0].transpose(0, 1)

    mean_abs_diff = (output_sdpa - flashinfer_output).abs().mean()
    allclose = torch.allclose(output_sdpa, flashinfer_output, atol=0.125, rtol=0)
    logger.info(f'Layer {layer_idx}, mean abs diff: {mean_abs_diff}, allclose: {allclose}')


# Register with HuggingFace Transformers
AttentionInterface.register("ecahlang_attention", ecahlang_attention)
