"""Minimal LLaDA adapter for the diffusion tuned lens.

Only what a tuned lens needs: a frozen ``ln_f -> ff_out`` unembed, the block
list (for residual-stream readout), the model dims and the mask-token id.
No model_surgery / FSDP / DDP machinery -- see the parent tuned-lens repo if
that is ever required.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from jlens.diffusion_corpus import get_mask_token_id


def load_llada(
    name: str,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    token: str | None = None,
) -> tuple[nn.Module, Any]:
    """Load a LLaDA-family checkpoint (``AutoModel`` + ``trust_remote_code``)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token)
    model = (
        AutoModel.from_pretrained(name, trust_remote_code=True, dtype=dtype, token=token)
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    return model, tokenizer


class LLaDAUnembed(nn.Module):
    """Frozen final norm + LM head copied out of a LLaDA model.

    LLaDA stores them at ``model.model.transformer.{ln_f,ff_out}``, which the
    upstream ``tuned_lens.nn.Unembed`` cannot locate (``model_surgery`` only
    knows the vanilla HF families). The interface here mirrors the upstream
    ``Unembed`` (``final_norm`` / ``unembedding`` / ``forward``) so the rest of
    the lens code is agnostic to it.

    Mirrors the model's own readout exactly: honours ``weight_tying`` (use
    ``wte`` instead of ``ff_out``) and ``scale_logits`` (multiply by
    ``d_model**-0.5``), if the config sets them.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        transformer = hf_model.model.transformer
        cfg = hf_model.config
        self.final_norm = copy.deepcopy(transformer.ln_f)
        self.scale_logits = bool(getattr(cfg, "scale_logits", False))
        self.d_model = int(getattr(cfg, "d_model", None) or cfg.hidden_size)
        if getattr(cfg, "weight_tying", False):
            self._tied_weight = transformer.wte.weight
            self.unembedding = None
        else:
            self.unembedding = copy.deepcopy(transformer.ff_out)
        self.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.unembedding is not None:
            weight = self.unembedding.weight
            x = self.final_norm(h.to(weight.dtype).to(weight.device))
            logits = self.unembedding(x)
        else:
            weight = self._tied_weight
            x = self.final_norm(h.to(weight.dtype).to(weight.device))
            logits = torch.nn.functional.linear(x, weight, None)
        if self.scale_logits:
            logits = logits * (self.d_model**-0.5)
        return logits


def model_dims(hf_model: nn.Module) -> tuple[int, int]:
    """``(n_layers, d_model)`` for either LLaDA (``n_layers``/``d_model``) or
    vanilla HF configs (``num_hidden_layers``/``hidden_size``)."""
    cfg = hf_model.config
    n_layers = getattr(cfg, "n_layers", None) or cfg.num_hidden_layers
    d_model = getattr(cfg, "d_model", None) or cfg.hidden_size
    return int(n_layers), int(d_model)


def resolve_mask_token_id(hf_model: nn.Module, *, fallback: int | None = None) -> int:
    """Mask-token id from the checkpoint config, with an explicit fallback
    (e.g. ``jlens.diffusion_corpus.ILLADA_MASK_ID`` for iLLaDA)."""
    return get_mask_token_id(hf_model, fallback=fallback)
