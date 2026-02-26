"""
Model runner: loads HuggingFace model with custom attention, provides forward pass.

Inspired by:
- vLLM v1/worker/gpu_model_runner.py
- SGLang model_executor/model_runner.py
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import to trigger AttentionInterface.register()
from . import attention  # noqa: F401

logger = logging.getLogger(__name__)


class ModelRunner:
    """
    Loads a HuggingFace CausalLM with FlashInfer attention and provides
    forward pass + decode helpers.
    """

    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.eos_token_id = None
        self.config = None

        # Model config values (populated after load)
        self.num_layers = None
        self.num_heads = None
        self.num_kv_heads = None
        self.head_dim = None
        self.vocab_size = None

    def load(self):
        """Load tokenizer and model onto GPU."""
        logger.info(f'Loading model: {self.args.model}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            attn_implementation="ecahlang_attention",
            torch_dtype=self.args.model_dtype,
        ).eval().cuda()

        # EOS token(s)
        eos = self.model.generation_config.eos_token_id
        if not isinstance(eos, list):
            eos = [eos]
        self.eos_token_id = torch.tensor(eos).cuda()

        # Extract model config
        config = self.model.config
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_kv_heads = getattr(
            config, "num_key_value_heads",
            config.num_attention_heads // getattr(config, "num_key_value_groups", 1),
        )
        self.head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        logger.info(
            f'Model loaded: layers={self.num_layers}, heads={self.num_heads}, '
            f'kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, '
            f'vocab={self.vocab_size}'
        )

    def forward(self, input_ids, position_ids, **kwargs):
        """Run model forward pass (prefill path)."""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )

    def decode_forward(self, input_ids, position_ids, **kwargs):
        """
        Run model forward pass (decode path).
        This is the function that gets torch.compiled or CUDA-graphed.
        """
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )

    def setup_torch_compile(self):
        """Wrap decode_forward with torch.compile."""
        logger.info(f'Compiling decode with mode={self.args.torch_compile_mode}')
        self.decode_forward = torch.compile(
            self.decode_forward,
            mode=self.args.torch_compile_mode,
            dynamic=True,
        )

    def tokenize(self, text):
        """Encode text to tensor."""
        return self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[0]

    def apply_chat_template(self, messages):
        """Apply chat template and return prompt string."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def batch_decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(token_ids)
