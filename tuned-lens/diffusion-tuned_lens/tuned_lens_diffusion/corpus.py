"""Diffusion-trajectory corpus for the tuned lens.

Thin layer over ``jlens.diffusion_corpus``: real LLaDA sampler trajectories
(one partially-masked snapshot per denoising step) plus the mask-position
selector that tuned-lens trains against. Only ``[MASK]`` positions are used
as targets -- a still-masked position is exactly where the model is being
asked to commit content, and therefore where the final-layer distribution is
a meaningful distillation target. This mirrors jlens' target-mask choice, but
tuned-lens needs no separate source mask (it is position-wise, not averaged).
"""

from __future__ import annotations

import torch

# Re-exported so callers have a single import site.
from jlens.diffusion_corpus import (  # noqa: F401
    ILLADA_MASK_ID,
    MASK_ID,
    build_diffusion_examples,
    build_single_mask_examples,
    get_mask_token_id,
)


def masked_positions(input_ids: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Boolean ``[seq_len]`` mask of still-masked positions (training targets).

    A snapshot may have zero such positions (the fully-revealed final step of
    a trajectory); callers skip those.
    """
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    return (ids == mask_id).cpu()


def build_corpus(
    hf_model,
    tokenizer,
    prompts_text: list[str],
    *,
    device: str | torch.device,
    mask_id: int,
    **kwargs,
) -> tuple[list[torch.Tensor], list[int]]:
    """``build_diffusion_examples`` + drop snapshots that have no ``[MASK]``.

    Returns ``(examples, tau_bins)`` -- the masks are recomputed from
    ``input_ids`` on the fly by :func:`masked_positions`, so there is nothing
    else to carry around.

    Extra ``kwargs`` are forwarded to ``build_diffusion_examples`` (e.g.
    ``total_len``, ``prompt_lens``, ``n_prompt_lens_per_text``, ``n_bins``).
    """
    examples, _source_masks, _target_masks, tau_bins = build_diffusion_examples(
        hf_model, tokenizer, prompts_text, device=device, mask_id=mask_id, **kwargs
    )
    keep = [i for i, ex in enumerate(examples) if masked_positions(ex, mask_id).any()]
    return [examples[i] for i in keep], [tau_bins[i] for i in keep]
