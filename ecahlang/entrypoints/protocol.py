from pydantic import BaseModel
from typing import List, Optional


class Parameters(BaseModel):
    model: str = 'model'
    temperature: float = 1.0
    top_p: float = 0
    top_k: int = 0
    max_tokens: int = 256
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    stream: bool = False


class ChatMessage(BaseModel):
    role: Optional[str] = 'user'
    content: Optional[str] = 'Hello!'


class ChatCompletionForm(Parameters):
    messages: List[ChatMessage] = [{'role': 'user', 'content': 'Hello!'}]


class CompletionForm(Parameters):
    prompt: str = 'Hello!'
