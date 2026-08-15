"""PicoLM — a GPT-style language model built from scratch in PyTorch.

Includes a hand-written byte-pair-encoding (BPE) tokenizer, a causal
transformer, a GPU-accelerated training loop with mixed precision, KV-cache
inference, int8 weight quantization, and a Streamlit demo.

Highlights::

    from picolm import GPT, ModelConfig
    model = GPT(ModelConfig(vocab_size=65, block_size=256))
"""

from picolm.model import GPT, ModelConfig
from picolm.tokenizer import BPETokenizer, CharTokenizer
from picolm._version import __version__

__all__ = [
    "GPT",
    "ModelConfig",
    "CharTokenizer",
    "BPETokenizer",
    "__version__",
]
