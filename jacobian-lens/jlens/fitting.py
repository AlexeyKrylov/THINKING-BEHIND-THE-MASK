# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:func:`jacobian_for_prompt`): for each output dimension, inject a
one-hot cotangent at *every valid target position at once* and backprop. The
gradient at source position ``p`` is then ``sum_{p'} dh_final[p'] / dh_l[p]``,
summed over target positions ``p'``; we take the mean over source positions
``p``. This is the reduction used in the paper. A per-position estimator
(``dh_final[p] / dh_l[p]`` averaged over ``p``) gives a slightly different
``J_l``; both work as a lens.

# CHANGED (doc only): clarified why nothing here needs a "bidirectional mode".
For strictly causal models the sum over target positions is implicitly
restricted to ``p' >= p`` -- attention can't look forward, so gradients from
earlier targets vanish on their own; that restriction was never encoded in
this file, it falls out of the model's own attention mask. For non-causal /
bidirectional models (e.g. diffusion LMs such as LLaDA) the exact same code
sums over *all* target positions in both directions, which is exactly what we
want there. So: no code change was needed for "bidirectional" as such.

# NEW (doc only): what *does* need to change, and why.
What does need to change for diffusion snapshots is which positions count as
a source vs. a target within a single example: for a partially-revealed
diffusion state, a still-masked position hasn't committed to any content yet,
so it shouldn't be averaged over as a *source* of translatable information
(though it remains a perfectly good *target*: "what is the model currently
leaning towards putting there" is exactly the interesting signal for
multi-hop reasoning). See :func:`revealed_position_mask` and the new
``source_position_mask`` / ``target_position_mask`` arguments below.

Cost: one forward pass and ``ceil(d_model / dim_batch)`` backward passes per
example. Shard across machines by running :func:`fit` on disjoint slices and
merging with :meth:`jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
# CHANGED (comment only): flagging that this constant was calibrated on
# causal autoregressive models. Nothing here forces you to change the value,
# but don't assume it transfers unchecked to a bidirectional model, where
# early positions need not play the same attention-sink role. Recommend
# empirically re-checking (e.g. per-position activation-norm profile) before
# trusting it for LLaDA.
SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    seq_len: int,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    # NEW: exclude_last is a new parameter. Reason: the original hardcoded
    # `seq_len - 1` exclusion assumes a causal LM, where the final position
    # has no next-token target. A bidirectional/diffusion model has no such
    # privileged final position, so that exclusion needs to be switchable
    # rather than baked in.
    exclude_last: bool = True,
) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour, so they're
    excluded. For a causal LM the final position also has no next-token
    target and is excluded by default (``exclude_last=True``, the original
    behaviour). For a bidirectional / diffusion model the final position
    isn't special in that way, so pass ``exclude_last=False`` there.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.
        exclude_last: Whether to drop the final position. Defaults to
            ``True``, matching the original (causal-model) behaviour exactly.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short to
            leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    # CHANGED: `end` used to be hardcoded to `seq_len - 1`. Now parameterized
    # so exclude_last=False (diffusion case) keeps the final position.
    end = seq_len - 1 if exclude_last else seq_len
    mask[skip_first:end] = True
    if mask.sum() == 0:
        # CHANGED: error message adjusted to reflect the parameterized `end`
        # (previously always implied "+1" for the excluded last position).
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > "
            f"{skip_first + (1 if exclude_last else 0)} tokens"
        )
    return mask


# NEW: entire function. Reason: for a diffusion snapshot, some positions are
# still `[MASK]` -- content the model hasn't committed to. Treating a masked
# position as a *source* of the Jacobian average would mix in the gradient of
# essentially uninitialized content, which is why the source-position mask
# needs to differ from the target-position mask for these examples (see the
# module docstring). This mirrors valid_position_mask but additionally
# intersects with "not currently masked".
def revealed_position_mask(
    input_ids: torch.Tensor,
    mask_token_id: int,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    exclude_last: bool = False,
) -> torch.Tensor:
    """Source-position mask for a partially-revealed diffusion snapshot.

    Intersects :func:`valid_position_mask` with "not currently masked":
    a masked position is content the model hasn't committed to yet, so it
    isn't a meaningful *source* of translatable information (it's still a
    fine *target*; see the module docstring).

    Args:
        input_ids: Tensor of shape ``[1, seq_len]`` (or ``[seq_len]``) holding
            the token ids of one diffusion snapshot, batch dim optional.
        mask_token_id: The model's mask-token id (e.g. ``126336`` for LLaDA).
        skip_first: See :func:`valid_position_mask`.
        exclude_last: See :func:`valid_position_mask`. Defaults to ``False``
            here since diffusion models have no privileged "final position".

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If no positions remain after masking and ``skip_first``.
    """
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    seq_len = ids.shape[-1]
    base = valid_position_mask(seq_len, skip_first=skip_first, exclude_last=exclude_last)
    revealed = (ids != mask_token_id).cpu()
    combined = base & revealed
    if combined.sum() == 0:
        raise ValueError(
            "no revealed positions left after applying mask_token_id and skip_first"
        )
    return combined


def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    # UNCHANGED from the original file.
    """Resolve None/negative layer indices, bounds-check, enforce source < target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(
            f"target_layer={target_layer} out of range for {n_layers} layers"
        )
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    # CHANGED: `prompt` is now `str | None` (was `str`). Reason: needs to be
    # optional so a pre-tokenized diffusion snapshot can be passed via
    # `input_ids` instead, without tokenizing text that doesn't exist.
    prompt: str | None,
    source_layers: Sequence[int],
    *,
    # NEW: lets callers pass an already-tokenized example (e.g. a diffusion
    # snapshot containing [MASK] tokens) instead of text. Reason: such
    # snapshots aren't plain text `model.encode` should tokenize from
    # scratch -- they're intermediate generation states.
    input_ids: torch.Tensor | None = None,
    # NEW: see module docstring -- source and target positions can differ
    # for diffusion snapshots (masked positions are valid targets but not
    # valid sources).
    source_position_mask: torch.Tensor | None = None,
    target_position_mask: torch.Tensor | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one example.

    Runs one forward pass on the input replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid *target* position. Rows
    are then averaged over valid *source* positions. See the module
    docstring for the resulting estimator, how it relates to a strict
    per-position Jacobian, and why source and target positions can differ
    (diffusion snapshots).

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text. Mutually exclusive with ``input_ids``; exactly
            one of the two must be given.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        input_ids: Pre-tokenized input of shape ``[1, seq_len]``, e.g. a
            partially-revealed diffusion snapshot containing mask tokens.
            Use this instead of ``prompt`` when the input isn't plain text
            that ``model.encode`` should tokenize from scratch.
        source_position_mask: Boolean mask of shape ``[seq_len]`` selecting
            which positions act as the *source* ``h_l[p]`` to average the
            gradient over. Defaults to ``target_position_mask`` (the original
            behaviour, appropriate for plain text where every valid position
            is both a legitimate source and target).
        target_position_mask: Boolean mask of shape ``[seq_len]`` selecting
            which positions get the injected cotangent (the *target*
            ``h_final[p']`` positions summed over). Defaults to
            :func:`valid_position_mask` with ``skip_first``.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens. Only used when
            tokenizing from ``prompt``; ignored when ``input_ids`` is given.
        skip_first: Leading positions to exclude when a mask isn't given
            explicitly; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
        ``n_valid_positions`` is the number of source positions averaged over.

    Raises:
        ValueError: If neither or both of ``prompt``/``input_ids`` are given,
            or (via the mask helpers) if no valid positions remain.
    """
    # NEW: enforce the new "exactly one of prompt / input_ids" contract.
    if (prompt is None) == (input_ids is None):
        raise ValueError("exactly one of `prompt` or `input_ids` must be given")

    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    # CHANGED: tokenization is now conditional -- only runs when `input_ids`
    # wasn't already supplied. Previously this line unconditionally called
    # model.encode(prompt, ...).
    if input_ids is None:
        input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]

    # CHANGED: mask construction now goes through the new
    # target_position_mask / source_position_mask arguments instead of a
    # single `position_mask` used for both roles.
    if target_position_mask is None:
        target_position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    if source_position_mask is None:
        # Original behaviour, preserved exactly: source positions == target
        # positions when no explicit source mask is given (plain-text case).
        source_position_mask = target_position_mask

    n_valid_positions = int(source_position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # UNCHANGED: one forward on the input replicated dim_batch times.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[
            target_layer
        ]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        # CHANGED: renamed from `valid_positions` to `target_positions` and
        # now derived from `target_position_mask` specifically (previously
        # the single `position_mask` served both roles implicitly).
        target_positions = target_position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        # NEW: separate source-position index list.
        source_positions = source_position_mask.nonzero(as_tuple=True)[0]

        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # UNCHANGED logic, just renamed variable: cotangent is still
            # placed at every valid *target* position (this is what encodes
            # "sum over t' >= t" for causal models, automatically, via the
            # model's own attention mask -- see module docstring).
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                target_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # CHANGED: averages over `source_positions` instead of the
                # old shared `valid_positions`. For plain text these are
                # identical (no behaviour change); for a diffusion snapshot
                # this now excludes still-masked positions from the average,
                # per the module docstring's rationale.
                positions_on_device = source_positions.to(grad.device, non_blocking=True)
                rows = (
                    grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                )
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    # UNCHANGED from the original file.
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    # CHANGED: type widened from `Sequence[str]` to `Sequence[str] |
    # Sequence[torch.Tensor]`. Reason: elements may now be plain text or
    # pre-tokenized diffusion snapshots (torch.Tensor), possibly mixed.
    prompts: Sequence[str] | Sequence[torch.Tensor],
    *,
    # NEW: per-example mask lists, same order/length as `prompts`. Reason:
    # this is how a diffusion trajectory corpus (one snapshot per sampling
    # step) supplies its source/target position masks; also how it gets
    # "averaging over diffusion steps" for free -- one call to fit() over
    # all collected snapshots is already the running mean over every step,
    # with no separate step-averaging mechanism required.
    source_position_masks: Sequence[torch.Tensor] | None = None,
    target_position_masks: Sequence[torch.Tensor] | None = None,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
    """Fit ``J_l`` over a list of examples and return a :class:`JacobianLens`.

    Per-example Jacobians from :func:`jacobian_for_prompt` are accumulated as
    a running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` examples (atomic) and resumed from on restart.

    Each element of ``prompts`` may be plain text (``str``, tokenized
    internally via ``model.encode``) or a pre-tokenized ``torch.Tensor`` of
    shape ``[1, seq_len]`` -- e.g. a partially-revealed diffusion snapshot
    containing mask tokens. Elements may be mixed within one call.

    To average over a diffusion sampling trajectory: collect one snapshot
    per sampling step (see ``diffusion_corpus.collect_trajectory`` /
    ``build_diffusion_examples``), pass each snapshot's ``input_ids`` as an
    element of ``prompts``, and pass the matching
    ``revealed_position_mask`` / ``valid_position_mask(...,
    exclude_last=False)`` in ``source_position_masks`` /
    ``target_position_masks``. One call to :func:`fit` then already averages
    over every step you included -- no separate "average over diffusion
    steps" mechanism is needed; it's the same running mean as averaging over
    prompts.

    Args:
        model: The model to fit on.
        prompts: Text prompts and/or pre-tokenized snapshots to average over.
            See the README for guidance on corpus size and distribution.
        source_position_masks: Per-example source-position masks, same
            length and order as ``prompts``. ``None`` at an index (or the
            whole argument) falls back to :func:`jacobian_for_prompt`'s
            default (source == target positions).
        target_position_masks: Per-example target-position masks, same
            length and order as ``prompts``. ``None`` falls back to
            :func:`valid_position_mask` with ``skip_first``.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each text prompt to this many tokens. Ignored
            for pre-tokenized elements.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N examples (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    # UNCHANGED from here down to the main loop.
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d examples",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-example Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short example that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d examples processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        # UNCHANGED.
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()

        # NEW: dispatch on whether this element is already tokenized, and
        # look up this example's masks (if a diffusion corpus supplied any).
        is_pretokenized = isinstance(prompt, torch.Tensor)
        src_mask = source_position_masks[prompt_idx] if source_position_masks else None
        tgt_mask = target_position_masks[prompt_idx] if target_position_masks else None

        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                # CHANGED: pass text only when NOT pre-tokenized; pass
                # input_ids only when it IS. Previously this call always
                # passed `prompt` as text.
                None if is_pretokenized else prompt,
                source_layers,
                input_ids=prompt if is_pretokenized else None,
                source_position_mask=src_mask,
                target_position_mask=tgt_mask,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping example %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # UNCHANGED: per-example diagnostics and running-mean update.
        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  example %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no examples were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d examples", n_done)
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
