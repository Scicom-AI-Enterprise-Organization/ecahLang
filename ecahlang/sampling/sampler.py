import torch
import flashinfer


def logits_to_probs(logits, mask_penalty, temperature, top_k, top_p):
    """
    Sample next tokens from logits using top-k, top-p, temperature, and repetition penalty.

    Uses FlashInfer's fused top-k/top-p sampling kernel for efficiency.

    Args:
        logits: [batch_size, vocab_size]
        mask_penalty: [batch_size, vocab_size] - repetition penalty mask
        temperature: [batch_size, 1] - sampling temperature
        top_k: [batch_size] - top-k values (int32)
        top_p: [batch_size] - top-p values (float32)

    Returns:
        token_ids: [batch_size, 1]
    """
    logits = logits / mask_penalty
    logits = logits / temperature

    token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k=top_k, top_p=top_p, deterministic=True,
    )[None].T

    return token_ids


def logits_to_probs_torch(logits, mask_penalty, temperature, top_k, top_p):
    """
    Fallback sampling without FlashInfer (pure PyTorch).
    Uses Gumbel-max trick for sampling.
    """
    logits = logits / mask_penalty
    logits = logits / temperature

    topk_vals, _ = torch.topk(logits, top_k.max(), dim=1)
    pivot = topk_vals.gather(1, (top_k - 1).unsqueeze(1))
    logits = torch.where(logits < pivot, torch.full_like(logits, -float('inf')), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True)


def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    """
    Build block-diagonal causal mask for multi-sequence prefill.
    Used for SDPA comparison/debug only.
    """
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0
    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)
