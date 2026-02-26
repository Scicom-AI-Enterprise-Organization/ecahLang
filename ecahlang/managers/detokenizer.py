"""
Background detokenizer: offloads tokenizer.batch_decode() to a thread pool.

tokenizer.batch_decode() is pure CPU work that blocks the async event loop.
By running it in a ThreadPoolExecutor, we keep the event loop responsive
for queue processing and client I/O.

Inspired by SGLang managers/detokenizer_manager.py
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2)


async def background_batch_decode(tokenizer, token_ids):
    """
    Decode token IDs to text strings in a background thread.

    Args:
        tokenizer: HuggingFace tokenizer
        token_ids: tensor of token IDs [batch_size, 1]

    Returns:
        list of decoded strings
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, tokenizer.batch_decode, token_ids,
    )
