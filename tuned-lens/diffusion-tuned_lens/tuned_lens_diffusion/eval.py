"""Evaluation / readout for the diffusion tuned lens.

Metrics are computed at still-masked positions (the same targets used in
training), against the model's own final-layer distribution -- the diffusion
analogue of the upstream tuned-lens per-layer KL. A logit-lens baseline
(identity translator) is included for comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch as th
from tqdm import tqdm

from .corpus import masked_positions


@th.no_grad()
def _forward(hf_model, input_ids, mask_id):
    out = hf_model(input_ids, output_hidden_states=True)
    hidden = out.hidden_states[:-1]
    final_lps = out.logits.float().log_softmax(dim=-1)
    targets = masked_positions(input_ids, mask_id).to(input_ids.device)
    return hidden, final_lps, targets


@th.no_grad()
def evaluate(hf_model, lens, examples, *, mask_id, device: str | th.device | None = None) -> dict:
    """Aggregate per-layer KL, entropy and top-1 agreement over ``examples``.

    Returns a dict with ``tuned`` / ``logit`` each holding ``kl``, ``entropy``
    and ``top1`` lists indexed by layer. ``top1`` is agreement with the model's
    final argmax. Note there is no layer that reproduces the model exactly:
    with ``hidden_states[:-1]`` the deepest readout is the input to the last
    block (not its output), so ``top1`` at the last layer stays below 1.0.
    ``entropy`` is in nats, matching upstream tuned-lens.
    """
    device = next(hf_model.parameters()).device if device is None else th.device(device)
    lens = lens.to(device)
    n_layers = lens.config.num_hidden_layers
    accum = {
        name: {m: th.zeros(n_layers) for m in ("kl", "entropy", "top1")}
        for name in ("tuned", "logit")
    }
    n = 0
    for ex in tqdm(examples):
        input_ids = ex.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden, final_lps, targets = _forward(hf_model, input_ids, mask_id)
        if not bool(targets.any()):
            continue
        final_p = final_lps.exp()
        final_argmax = final_lps.argmax(dim=-1)[0]
        pos = targets
        for i, h in enumerate(hidden):
            for name, lps in (
                ("tuned", lens(h, i).log_softmax(dim=-1)),
                ("logit", lens.unembed(h).log_softmax(dim=-1)),
            ):
                kl = th.sum(final_p * (final_lps - lps), dim=-1)[0][pos].mean()
                entropy = th.sum(-lps.exp() * lps, dim=-1)[0][pos].mean()
                top1 = (lps.argmax(dim=-1)[0][pos] == final_argmax[pos]).float().mean()
                accum[name]["kl"][i] += kl.cpu()
                accum[name]["entropy"][i] += entropy.cpu()
                accum[name]["top1"][i] += top1.cpu()
        n += 1
    if n == 0:
        raise ValueError("no examples had masked positions to evaluate on")
    return {
        name: {metric: (values / n).tolist() for metric, values in metrics.items()}
        for name, metrics in accum.items()
    }


@th.no_grad()
def lens_logits(
    hf_model,
    lens,
    *,
    input_ids: th.Tensor | None = None,
    prompt: str | None = None,
    tokenizer=None,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 512,
    device: str | th.device | None = None,
):
    """Per-layer lens logits at ``positions`` (jlens ``apply``-style contract).

    Returns ``(lens_logits, model_logits, input_ids)`` where ``lens_logits`` is
    ``{layer: [n_positions, vocab]}``. No [MASK] is appended here: pass the
    exact ``input_ids`` you want read (e.g. real context + one trailing mask).
    """
    device = next(hf_model.parameters()).device if device is None else th.device(device)
    lens = lens.to(device)
    if (prompt is None) == (input_ids is None):
        raise ValueError("exactly one of `prompt` or `input_ids` must be given")
    if input_ids is None:
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).input_ids
    input_ids = input_ids.to(device)

    out = hf_model(input_ids, output_hidden_states=True)
    hidden = out.hidden_states[:-1]
    final_logits = out.logits.float()
    if layers is None:
        layers = range(lens.config.num_hidden_layers)

    def select(x):
        x = x[0]
        return x if positions is None else x[list(positions)]

    out_lens = {i: select(lens(hidden[i], i).float()).cpu() for i in layers}
    model_logits = select(final_logits).cpu()
    return out_lens, model_logits, input_ids
