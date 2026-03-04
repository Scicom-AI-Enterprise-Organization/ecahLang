from pydantic import BaseModel
from typing import Union, List, Optional

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

    def __str__(self) -> str:
        if self.role == 'system':
            return f'system:\n{self.content}\n'

        elif self.role == 'user':
            if self.content is None:
                return 'user:\n</s>'
            else:
                return f'user:\n</s>{self.content}\n'

        elif self.role == 'assistant':

            if self.content is None:
                return 'assistant'

            else:
                return f'assistant:\n{self.content}\n'

        else:
            raise ValueError(f'Unsupported role: {self.role}')

class ChatCompletionForm(Parameters):
    messages: List[ChatMessage] = [{'role': 'user', 'content': 'Hello!'}]

class CompletionForm(Parameters):
    prompt: str = 'Hello!'
    