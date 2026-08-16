"""Model configuration.

A single dataclass controls the architecture, so the same code scales from a
sub-mega-parameter toy model (trainable on a laptop in seconds) up to
GPT-2-scale architectures (given the compute).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """GPT architecture hyper-parameters.

    Follows the GPT-2 naming convention so it maps 1:1 onto the original
    paper's notation (``n_layer``, ``n_head``, ``n_embd``).
    """

    vocab_size: int = 65          # number of tokens in the vocabulary
    block_size: int = 256         # maximum context length (T)
    n_layer: int = 6              # transformer blocks
    n_head: int = 6               # attention heads (must divide n_embd)
    n_embd: int = 384             # embedding / residual dimension
    dropout: float = 0.0          # dropout probability (0 for inference)
    bias: bool = False            # use bias in LayerNorm/linear layers (GPT-2: no)
    rmsnorm: bool = False         # use RMSNorm instead of LayerNorm (LLaMA-style)
    rope: bool = False            # rotary position embeddings instead of learned wpe
    grad_checkpoint: bool = False # recompute block activations in backward (memory/speed tradeoff)

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_params(self) -> int:
        """Return the total number of trainable parameters."""
        # token + position embeddings
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        per_block = 0
        d = self.n_embd
        # attention: q,k,v,proj -> 4 * d^2
        per_block += 4 * d * d
        # mlp: 4d->d->4d -> 8 * d^2
        per_block += 8 * d * d
        # final layernorm + lm_head (weight-tied to token embedding)
        per_block += 2 * d
        return emb + self.n_layer * per_block + d  # + d for final ln (approx)

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_head > self.n_embd:
            raise ValueError("n_head cannot exceed n_embd")


# Default "pico" model — trainable on a laptop GPU in a couple of minutes.
PICO_CONFIG = ModelConfig(
    vocab_size=65,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)
