"""Training loop for the diffusion tuned lens.

For each snapshot we run LLaDA once (no grad) to get the pre-block residuals
and the final-layer distribution, then distil the latter into every layer's
translator with a KL loss computed **only at still-masked positions**. There
is no token shift (a diffusion LM predicts each position from itself, not the
next one) and no source-position averaging (the lens is position-wise).

Per tau-bin (diffusion noise level) a separate lens is fit, then all bins can
be merged with :meth:`DiffusionTunedLens.merge` -- the tuned-lens analogue of
jlens' bin lenses + merge.

Like upstream tuned-lens, fitting is a **single pass** over the corpus -- one
optimizer step per snapshot (batch size 1), no epochs (upstream is step-based:
``num_steps`` = number of steps, and its dataloader is never cycled). The
training budget is therefore controlled by how many snapshots the caller
passes in (e.g. a stratified subsample), not by an epoch count.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from tqdm import tqdm

import torch as th

from .corpus import masked_positions
from .lenses import DiffusionTunedLens

logger = logging.getLogger(__name__)


@th.no_grad()
def _forward(hf_model, input_ids, mask_id):
    """One LLaDA forward: ``(hidden_states[:-1], final_log_probs, target_mask)``.

    ``hidden_states[:-1]`` has one entry per transformer block (the residual
    fed into that block); the last HF entry (post-``ln_f``) is the model output
    and is represented by ``final_log_probs`` instead.
    """
    out = hf_model(input_ids, output_hidden_states=True)
    hidden = out.hidden_states[:-1]
    final_lps = out.logits.float().log_softmax(dim=-1)
    targets = masked_positions(input_ids, mask_id).to(input_ids.device)
    return hidden, final_lps, targets


def train_one_bin(
    hf_model,
    examples: list[th.Tensor],
    *,
    mask_id: int,
    tau_bin: int | None = None,
    bias: bool = True,
    device: str | th.device = "cuda",
    lr: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    log_every: int = 1,
) -> DiffusionTunedLens:
    """Fit one :class:`DiffusionTunedLens` over a list of snapshots.

    Single pass over ``examples`` (one optimizer step each), matching the
    upstream tuned-lens step-based loop -- no epochs.

    Args:
        hf_model: Loaded LLaDA model (eval, frozen, on ``device``).
        examples: Pre-tokenized snapshots, each ``[1, seq_len]``.
        mask_id: The model's mask-token id.
        tau_bin: Tag stored on the resulting lens (diffusion-stage bookkeeping).
        bias: Whether translators have a bias.
        device: Device to train on.
        lr, momentum, weight_decay: SGD (same residual-parametrization defaults
            as upstream tuned-lens: LR is effectively ``lr`` after undoing
            PyTorch's ``1/(1-momentum)`` scaling).
        grad_clip: Max grad norm.
        log_every: Log every N successful examples.

    Returns:
        The fitted lens, with ``config.n_prompts`` set to the number of
        examples actually used (those with at least one masked position).
    """
    lens = DiffusionTunedLens.from_model(hf_model, bias=bias, tau_bin=tau_bin).to(device)
    params = [p for p in lens.parameters() if p.requires_grad]
    opt = th.optim.SGD(
        params,
        lr=lr * (1 - momentum),
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )

    n_seen = 0
    running = defaultdict(list)
    for ex in examples:
        input_ids = ex.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden, final_lps, targets = _forward(hf_model, input_ids, mask_id)
        if not bool(targets.any()):
            continue

        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        layer_losses = []
        final_p = final_lps.exp()
        for i, h in enumerate(hidden):
            lps = lens(h, i).log_softmax(dim=-1)
            kl = th.sum(final_p * (final_lps - lps), dim=-1)[0]  # [seq_len]
            loss = kl[targets].mean()
            loss.backward()
            layer_losses.append(loss.detach().item())
        th.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        n_seen += 1
        for i, value in enumerate(layer_losses):
            running[i].append(value)

        if log_every and n_seen % log_every == 0:
            logger.info(
                "  [tau=%s] example %d  KL: L0=%.3f L%d=%.3f  (%.2fs)",
                tau_bin,
                n_seen,
                layer_losses[0],
                len(layer_losses) - 1,
                layer_losses[-1],
                time.perf_counter() - start,
            )

    lens.config.n_prompts = n_seen
    if n_seen:
        per_layer = {i: sum(v) / len(v) for i, v in running.items()}
        logger.info(
            "  [tau=%s] done: %d examples, mean KL L0=%.3f -> L%d=%.3f",
            tau_bin,
            n_seen,
            per_layer[0],
            max(per_layer),
            per_layer[max(per_layer)],
        )
    return lens


def train_bins(
    hf_model,
    examples: list[th.Tensor],
    tau_bins: list[int],
    *,
    mask_id: int,
    **kwargs,
) -> dict[int, DiffusionTunedLens]:
    """Fit one lens per distinct tau-bin value. `kwargs` -> :func:`train_one_bin`."""
    bin_lenses: dict[int, DiffusionTunedLens] = {}
    for b in tqdm(sorted(set(tau_bins))):
        idx = [i for i, bb in enumerate(tau_bins) if bb == b]
        logger.info("--- tau bin %s: %d snapshots ---", b, len(idx))
        bin_lenses[b] = train_one_bin(
            hf_model, [examples[i] for i in idx], mask_id=mask_id, tau_bin=b, **kwargs
        )
    return bin_lenses
