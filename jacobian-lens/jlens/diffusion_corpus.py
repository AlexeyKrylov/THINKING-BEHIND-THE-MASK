# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# NEW FILE: none of this existed before. It doesn't touch jlens internals --
# it only builds the (examples, source_masks, target_masks, tau_bins) corpus
# that the patched fitting.py already knows how to consume. Kept separate
# from fitting.py on purpose: LLaDA's sampling loop is model-specific
# (semi-autoregressive blocks + low-confidence remasking), while fitting.py
# stays generic across any LensModel.
"""Building a diffusion-trajectory corpus for jlens.fitting.fit on LLaDA.

This module does not touch jlens internals; it only produces the
``examples`` / ``source_position_masks`` / ``target_position_masks`` triple
(plus ``tau_bins`` for per-stage binning) that the patched
:func:`jlens.fitting.fit` (see ``fitting.py`` in this same patch) accepts
directly.

# CHANGED: build_diffusion_examples was reworked to match the paper's
# "pure diffusion" sampling regime (steps == gen_length == block_length, i.e.
# one token revealed per step, no semi-autoregressive blocking -- see
# Table 2/7/8 and Appendix B.6 of the LLaDA paper, where this is confirmed
# to be the regime behind the headline results, not the faster/blockier
# regimes shown separately in Fig. 5's throughput analysis), AND to sweep
# the prompt/response split (``prompt_lens``) rather than using a single
# fixed prompt length. Total sequence length (``total_len``) is fixed at
# 128 to match the plain-text lens's ``max_seq_len=128`` for a fair budget
# comparison between the two lenses.
#
# Rationale for sweeping prompt_lens instead of a single split: a fixed
# 16/112 split means the lens only ever sees "short prompt, long response"
# trajectories. Sweeping prompt_len across (nearly) the whole budget gives
# the lens exposure to many different context/response ratios -- but note
# this multiplies collection+fitting cost by roughly
# sum(total_len - p for p in prompt_lens) / (total_len - min(prompt_lens))
# relative to the single-split version (~56x for the full range(16, 128)
# default below). See the caller notebook for the post-hoc stratified
# subsampling step used to cap the *fitting* cost back down without
# reducing collection coverage.
#
# Rationale for the skip_first split (source vs target): the prompt is
# never masked at any point in the trajectory (see collect_trajectory), so
# treating it as a *target* would put the injected cotangent on positions
# the model was never trained to produce content at (LLaDA's training loss,
# Eq. 3 in the paper, is computed only where x_t^i = M) -- exactly the
# "standard approach" pitfall discussed at length in the accompanying
# conversation. So the prompt is EXCLUDED from target via a per-variant,
# DYNAMIC skip_first_target == prompt_len. But as a *source* the prompt is
# perfectly legitimate real content (never masked, always "committed"), so
# it should NOT be excluded just because it happens to be the prompt --
# only the same small, FIXED leading margin (skip_first_source) that would
# apply to any sequence, for the usual attention-sink reason. Using the
# prompt's own (varying) length for the source-side skip would silently
# exclude the ENTIRE prompt whenever skip_first_source happened to equal
# prompt_len -- this is a real edge case at the prompt_lens lower bound
# (prompt_len == skip_first_source == 16 below): at that single point the
# whole prompt is excluded from source too, since there's nothing left
# after skipping 16 of 16 positions. Left as-is deliberately (raising
# prompt_lens' lower bound to 17 would remove it, at the cost of one fewer
# variant) -- decide based on whether that single edge case matters for
# your analysis.

Usage sketch::

    from diffusion_corpus import build_diffusion_examples
    import jlens.fitting as fitting

    examples, src_masks, tgt_masks, tau_bins = build_diffusion_examples(
        hf_model, tokenizer, prompt_texts, device=device,
    )
    # NEW: optionally stratified-subsample down to a fitting-time budget
    # before calling fit() -- collection cost is unaffected either way,
    # since it already happened above. See the caller notebook for the
    # subsampling snippet (groups indices by tau_bins, then
    # random.Random(seed).sample(..., k) per bin).

    bin_lenses = {}
    for b in sorted(set(tau_bins)):
        idx = [i for i, bb in enumerate(tau_bins) if bb == b]
        bin_lenses[b] = fitting.fit(
            model,
            [examples[i] for i in idx],
            source_position_masks=[src_masks[i] for i in idx],
            target_position_masks=[tgt_masks[i] for i in idx],
            checkpoint_path=f"./llada_diffusion_ckpt_bin{b}.pt",
        )
        bin_lenses[b].save(f"./llada_diffusion_lens_bin{b}.pt")

    from jlens.lens import JacobianLens
    overall_lens = JacobianLens.merge(list(bin_lenses.values()))
    overall_lens.save("./llada_diffusion_lens_merged.pt")
"""

from __future__ import annotations

import random  # NEW: per-text random subsampling of prompt_lens (see build_diffusion_examples)

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm  # NEW: progress bar over (text, prompt_len) variants

from .fitting import revealed_position_mask, valid_position_mask

#: LLaDA's [MASK] token id (GSAI-ML/LLaDA-8B-Base and -Instruct, confirmed
#: directly from both checkpoints' config.json: "mask_token_id": 126336).
MASK_ID = 126336

# NEW: iLLaDA (GSAI-ML/iLLaDA-8B-Instruct) is a DIFFERENT model from LLaDA --
# different tokenizer (vocab_size=155136 vs 126464), different architecture
# (GQA: num_key_value_heads=8 vs 32, tied embeddings, different modeling
# class ILLaDAForCausalLM). Its [MASK] token is NOT the same id: confirmed
# from tokenizer_config.json's added_tokens_decoder, entry "5" ->
# "<[MASK]>". Always pass the mask_id matching the *actual* model you're
# sampling from -- silently reusing MASK_ID=126336 here would make
# revealed_position_mask() treat almost every position as "revealed" (since
# input_ids != 126336 is true almost everywhere), breaking the
# source/target split without raising any error.
ILLADA_MASK_ID = 5


# NEW: helper to avoid hardcoding a mask id at all when the checkpoint's own
# config exposes one. LLaDA-8B-Base/-Instruct do (config.mask_token_id ==
# 126336); iLLaDA's config.json does not expose one at all (it has to be
# read off the tokenizer's added_tokens_decoder instead, as done above for
# ILLADA_MASK_ID) -- which is exactly the kind of silent mismatch that
# caused the LLaDA -> iLLaDA mixup. Prefer this helper over a bare constant
# wherever a HF model/config object is available.
def get_mask_token_id(hf_model_or_config, *, fallback: int | None = None) -> int:
    """Reads ``mask_token_id`` off a HF model or its ``.config``, if present.

    Args:
        hf_model_or_config: Either a HF ``PreTrainedModel`` (its ``.config``
            is used) or a config object directly.
        fallback: Value to return if the config doesn't expose
            ``mask_token_id`` (e.g. iLLaDA). If ``None`` and no id is found,
            raises instead of silently guessing.

    Returns:
        The mask-token id.

    Raises:
        ValueError: If no ``mask_token_id`` is found and ``fallback`` is
            ``None``.
    """
    config = getattr(hf_model_or_config, "config", hf_model_or_config)
    mask_id = getattr(config, "mask_token_id", None)
    if mask_id is not None:
        return mask_id
    if fallback is not None:
        return fallback
    raise ValueError(
        "this checkpoint's config has no mask_token_id; pass `fallback=` "
        "explicitly (e.g. ILLADA_MASK_ID for iLLaDA models)"
    )


@torch.no_grad()
def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


@torch.no_grad()
def collect_trajectory(
    hf_model,
    prompt_ids: torch.Tensor,
    *,
    steps: int = 32,
    gen_length: int = 64,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = MASK_ID,
    device: str = "cuda",
) -> list[dict]:
    """Runs LLaDA's real semi-autoregressive block sampler and returns one
    snapshot per sampling step.

    Each snapshot is the actual partially-revealed state the model produces
    during real generation (block-structured, low-confidence remasking) --
    not an i.i.d. random mask, which the model never encounters this way at
    inference time.

    Args:
        hf_model: The raw HF LLaDA model (not the jlens ``LensModel`` wrapper;
            we need plain ``.logits`` output here).
        prompt_ids: Tensor of shape ``[1, prompt_len]``.
        steps: Total sampling steps across all blocks. Pass
            ``steps == gen_length`` for the paper's "pure diffusion" regime
            (one token revealed per step); ``build_diffusion_examples``
            below always does this.
        gen_length: Number of tokens to generate.
        block_length: Size of each semi-autoregressive block. Must divide
            ``gen_length``. Pass ``block_length == gen_length`` (a single
            block) for "pure diffusion"; smaller values give the
            semi-autoregressive "Block Diffusion LLaDA" regime instead (see
            module-level rationale in ``build_diffusion_examples``).
        temperature: Gumbel-noise temperature for token selection (``0`` =
            greedy, matching Algorithm 4/5 of the paper).
        mask_id: The model's mask-token id.
        device: Device to run on.

    Returns:
        A list of ``{"input_ids", "frac_masked", "block", "step"}`` dicts, one
        per sampling step, in trajectory order. ``frac_masked`` is the
        fraction of masked tokens within the RESPONSE portion only (see
        below) -- not the whole sequence.
    """
    prompt_len = prompt_ids.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt_ids.to(device)

    snapshots: list[dict] = []
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for b in range(num_blocks):
        block_start = prompt_len + b * block_length
        block_end = prompt_len + (b + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if mask_index.sum() == 0:
                break

            # This is the real LLaDA sampler, not a synthetic i.i.d. mask:
            # low-confidence tokens stay masked and high-confidence ones get
            # committed each step. That block/confidence structure is exactly
            # what earlier discussion flagged as missing from a naive
            # "randomly mask each token independently" corpus -- the model
            # never actually encounters i.i.d. masks at inference time.
            logits = hf_model(x).logits
            logits_noisy = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=x.device)
            )
            confidence[:, block_end:] = -float("inf")  # only commit within this block

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = int(num_transfer_tokens[0, i].item())
            if k > 0:
                _, idx = torch.topk(confidence[0], k=k)
                transfer_index[0, idx] = True

            x = torch.where(transfer_index, x0, x)

            # Snapshot this step's actual state. Collecting one of these per
            # step, across many prompts, and handing the whole list to fit()
            # is what gives "averaging over diffusion steps" -- fit() already
            # does a running mean over whatever list of examples it's given,
            # so no separate step-averaging code is needed on the fitting
            # side; it only needed a real, non-i.i.d. corpus to average over.
            snapshots.append(
                {
                    "input_ids": x.clone(),
                    # CHANGED: fraction masked within the RESPONSE portion
                    # only (x[:, prompt_len:]), not the whole sequence. The
                    # prompt is never masked, so dividing by the full
                    # sequence length dilutes frac_masked by a factor of
                    # gen_length/seq_len < 1 -- this used to collapse
                    # adjacent steps into the same tau_bin under coarse
                    # n_bins and, worse, made bin counts uneven (verified:
                    # [16,32,32,32] steps/bin at n_bins=4 with the old
                    # formula vs. an even [28,28,28,28] with this one, for
                    # gen_length=112). frac_masked now ranges over the full
                    # [0, 1] interval regardless of prompt_len, which is
                    # also what makes tau_bins comparable across the
                    # different prompt_len variants swept in
                    # build_diffusion_examples below.
                    "frac_masked": (x[:, prompt_len:] == mask_id).float().mean().item(),
                    "block": b,
                    "step": i,
                }
            )

    return snapshots


def build_diffusion_examples(
    hf_model,
    tokenizer,
    prompts_text: list[str],
    *,
    device: str,
    # CHANGED: steps/gen_length/block_length are no longer independent
    # top-level arguments. They used to default to 32/64/32 (2
    # tokens/step, semi-autoregressive with 2 blocks), which does not match
    # any regime the LLaDA paper reports headline numbers for -- Table
    # 2/7/8 and Appendix B.6 confirm "pure diffusion" (one token per step,
    # no blocking) is what's behind the paper's main results, and Fig. 5's
    # faster-but-worse regimes are explicitly an efficiency/quality
    # trade-off, not the default. Pure diffusion is now hardcoded (each
    # collect_trajectory call below uses steps == gen_length ==
    # block_length), so there's no longer a way to accidentally regress to
    # the unfaithful 2-tokens/step regime through this function.
    total_len: int = 128,
    # NEW: sweep the prompt/response split instead of a single fixed
    # prompt length. Default is the full range 16..127 (112 variants) --
    # see the module-level rationale comment for the ~56x collection-cost
    # multiplier this implies relative to a single split, and for the
    # skip_first_source==16==prompt_lens.start edge case at the lower
    # bound. Pass a coarser range (e.g. range(16, 128, 8)) for a cheaper
    # sweep, or a single-element range/list to reproduce the old
    # single-split behaviour.
    prompt_lens: range | list[int] = range(16, 128),
    # NEW: instead of sweeping every prompt_len for every text (N * 112
    # variants -- e.g. 31h wall-clock at ~1 it/s for N=1000, before any
    # fitting even starts), draw a random SUBSET of prompt_lens per text.
    # Unlike the post-hoc stratified subsampling shown in the module
    # docstring's usage sketch (which only cuts fit()'s cost -- collection
    # still runs the full sweep first), this cuts BOTH collection and
    # fitting cost, since the un-sampled variants are never generated at
    # all. Each text gets its OWN independent random subset (not the same
    # subset for every text), so the corpus still sees the full range of
    # prompt_lens overall, just sparsely per individual text rather than
    # densely. ``None`` (default) keeps the old behaviour: every prompt_len
    # for every text.
    n_prompt_lens_per_text: int | None = None,
    # NEW: seed for the per-text prompt_lens sampling above. ``None`` uses
    # unseeded (system) randomness -- pass an int for a reproducible corpus
    # across reruns (independent of any global `random.seed(...)` set
    # elsewhere in the notebook).
    sample_seed: int | None = None,
    # NEW: fixed leading margin excluded from the SOURCE mask on every
    # variant, regardless of prompt_len -- this is the plain attention-sink
    # margin (same rationale/value as SKIP_FIRST_N_POSITIONS in fitting.py),
    # deliberately NOT tied to where the prompt happens to end. See the
    # module-level rationale for why source and target need independent
    # skip values here.
    skip_first_source: int = 16,
    n_bins: int = 4,
    mask_id: int = MASK_ID,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Builds a ``(examples, source_masks, target_masks, tau_bins)`` corpus
    ready to hand to the patched ``jlens.fitting.fit``, sweeping the
    prompt/response split under a fixed total-sequence-length budget.

    For each prompt length ``p`` in ``prompt_lens``, tokenizes each input
    text, truncates to ``p`` tokens as the prompt, and runs a full pure-
    diffusion trajectory (``steps == gen_length == block_length ==
    total_len - p``) to generate the remaining ``total_len - p`` tokens.
    Every sampling step of every variant of every prompt becomes one
    example in the returned corpus.

    Args:
        hf_model: Raw HF LLaDA model (used for sampling; needs ``.logits``).
        tokenizer: Tokenizer for ``prompts_text``.
        prompts_text: Seed prompts to generate from. Texts that tokenize
            shorter than ``total_len`` are skipped (there isn't enough
            budget left to try the longest ``prompt_lens`` variant).
        device: Device to sample on.
        total_len: Fixed prompt+response token budget for every variant --
            matches the plain-text lens's ``max_seq_len=128`` so the two
            lenses are fit on directly comparable sequence lengths.
        prompt_lens: Prompt lengths to sweep; ``total_len - p`` tokens are
            generated for each ``p``. Defaults to every integer from 16 to
            127 inclusive.
        n_prompt_lens_per_text: If set, draw this many prompt_lens values
            AT RANDOM (without replacement, freshly per text) instead of
            sweeping every value in ``prompt_lens`` for every text. Cuts
            both collection and fitting cost roughly by
            ``len(prompt_lens) / n_prompt_lens_per_text`` (e.g. 112/8 ≈ 14x
            for the full default range). ``None`` (default) sweeps
            everything, as before.
        sample_seed: Seed for the sampling above; ``None`` uses unseeded
            randomness.
        skip_first_source: Fixed number of leading positions excluded from
            every SOURCE mask, independent of ``prompt_len``. See the
            module-level rationale for why this must not simply be set to
            ``prompt_len`` (that would exclude the entire prompt from the
            source side, which -- unlike on the target side -- has no
            justification: the prompt is real, never-masked content and is
            just as legitimate a Jacobian source as the generated part).
        n_bins: Number of bins to split trajectory snapshots into by
            ``frac_masked`` (i.e. by diffusion "noise level" tau), for
            fitting separate per-stage lenses before optionally merging
            them. Comparable across ``prompt_lens`` variants since
            ``frac_masked`` is now normalized by response length, not by
            ``total_len`` (see ``collect_trajectory``).
        mask_id: The model's mask-token id.

    Returns:
        ``(examples, source_masks, target_masks, tau_bins)``, all the same
        length (one entry per collected snapshot across all prompts and all
        ``prompt_lens`` variants). Downstream stratified subsampling (by
        ``tau_bins``, down to a fixed fitting-time budget) is left to the
        caller -- see the module docstring's usage sketch.
    """
    examples: list[torch.Tensor] = []
    source_masks: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    tau_bins: list[int] = []

    # CHANGED: n_variants_per_text -- either the full sweep (len(prompt_lens))
    # or the per-text random subset size (n_prompt_lens_per_text), whichever
    # applies. This is what the progress bar total and per-skip advance
    # below are now computed from, instead of always len(prompt_lens).
    prompt_lens_list = list(prompt_lens)
    n_variants_per_text = (
        min(n_prompt_lens_per_text, len(prompt_lens_list))
        if n_prompt_lens_per_text is not None
        else len(prompt_lens_list)
    )
    rng = random.Random(sample_seed)

    # NEW: макро-прогресс по ВСЕМ вариантам (text, prompt_len) -- это
    # len(prompts_text) * n_variants_per_text итераций, каждая из которых
    # внутри себя запускает один вызов collect_trajectory. Даёт общий ETA
    # на весь сбор корпуса.
    n_variants = len(prompts_text) * n_variants_per_text
    pbar = tqdm(total=n_variants, desc="build_diffusion_examples: варианты (text, prompt_len)")

    for text in prompts_text:
        full_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=total_len
        ).input_ids
        if full_ids.shape[1] < total_len:
            # Not enough tokens to try the longest prompt_lens variant on
            # this text; skip rather than silently shrinking total_len for
            # only some texts (which would make sequence length -- and
            # hence the source/target budget -- inconsistent across the
            # corpus). Still advance the bar for all skipped variants, or
            # the total/ETA would drift from n_variants.
            pbar.update(n_variants_per_text)
            continue

        # CHANGED: either every prompt_len (old behaviour) or a fresh random
        # subset drawn independently FOR THIS TEXT -- so different texts
        # cover different parts of the prompt_lens range rather than all
        # texts sharing one arbitrary fixed subset.
        this_prompt_lens = (
            rng.sample(prompt_lens_list, n_variants_per_text)
            if n_prompt_lens_per_text is not None
            else prompt_lens_list
        )

        for prompt_len in this_prompt_lens:
            prompt_ids = full_ids[:, :prompt_len]
            gen_length = total_len - prompt_len

            # CHANGED: pure diffusion, hardcoded -- steps == gen_length ==
            # block_length == 1 block covering the whole response, one
            # token revealed per step. See rationale at the top of this
            # function.
            trajectory = collect_trajectory(
                hf_model,
                prompt_ids,
                steps=gen_length,
                gen_length=gen_length,
                block_length=gen_length,
                mask_id=mask_id,
                device=device,
            )
            for snap in trajectory:
                input_ids = snap["input_ids"]
                seq_len = input_ids.shape[1]
                # target_mask: every valid position except the prompt itself
                # (skip_first=prompt_len, DYNAMIC per variant) -- masked
                # response positions ARE included (they pass through
                # valid_position_mask, which does not check mask_id at all;
                # only revealed-ness is checked on the source side, below).
                # We want to read out what the model is currently leaning
                # towards putting at not-yet-revealed positions too (see
                # fitting.py's module docstring on why targets and sources
                # differ here). exclude_last=False: diffusion models have
                # no privileged "final position" the way a causal LM does,
                # so we don't drop it.
                tgt_mask = valid_position_mask(
                    seq_len, skip_first=prompt_len, exclude_last=False
                )
                # source_mask: FIXED leading margin (skip_first_source),
                # independent of prompt_len, intersected with "already
                # revealed" -- content the model has actually committed to,
                # and so a meaningful source of "translatable" information
                # for the Jacobian average. The prompt itself (beyond the
                # fixed margin) remains a valid source: it's real,
                # never-masked content just like any revealed response
                # token, so excluding it wholesale would throw away
                # legitimate signal for no reason tied to diffusion
                # specifically.
                src_mask = revealed_position_mask(
                    input_ids, mask_id, skip_first=skip_first_source
                )
                examples.append(input_ids)
                source_masks.append(src_mask)
                target_masks.append(tgt_mask)
                # tau_bin: which "noise level" stage of the trajectory this
                # snapshot belongs to, based on the fraction of the RESPONSE
                # still masked (see collect_trajectory). Comparable across
                # prompt_lens variants since it's normalized by response
                # length, not by total_len.
                tau_bins.append(min(int(snap["frac_masked"] * n_bins), n_bins - 1))
            pbar.update(1)  # NEW: один вариант (text, prompt_len) обработан целиком

    pbar.close()  # NEW
    return examples, source_masks, target_masks, tau_bins


# NEW: companion corpus builder that does NOT run collect_trajectory at all.
#
# Rationale: the eval-time readout protocol (see evaluation.ipynb's
# call_llada_lens) constructs its own input as "real, fully-revealed context
# + exactly one appended [MASK] token, nothing after it" and reads the lens
# right there -- but neither build_diffusion_examples' tau-binned corpus nor
# any single bin drawn from it actually contains this exact configuration:
#
#   - a snapshot with gen_length=1 never has a "still masked" state saved at
#     all (the only snapshot taken is AFTER that single position commits);
#   - a snapshot with gen_length>1 and exactly one remaining mask CAN exist,
#     but low-confidence remasking reveals tokens out of left-to-right order,
#     so that last mask is not guaranteed to sit right at the sequence's end
#     with nothing but real content to its left.
#
# So every existing bin is only an approximate match to what evaluation.ipynb
# actually feeds the lens (bin3 turned out closest empirically -- it pools
# snapshots with the highest remaining-mask fraction, which is qualitatively
# nearest to "one mask, otherwise real content" -- but "closest" is not
# "identical"). This function builds the identical configuration directly,
# without any model forward pass at all: no sampler is needed because we are
# not asking what the model WOULD generate, only how it reads a context we
# construct ourselves. Cost is therefore just tokenization (CPU, no GPU
# calls) -- collection here is essentially free; fit()'s backward-pass cost
# is unchanged and still dominates, so callers may still want to subsample
# before fit() if `len(prompts_text) * len(prompt_lens)` is large.
#
# Because every example has an identical structural role (one masked
# position, real content everywhere else) there is no analogue of tau_bins
# here -- the whole corpus is a single homogeneous class, fit with one
# fit() call, not four bin-lenses to merge.
def build_single_mask_examples(
    tokenizer,
    prompts_text: list[str],
    *,
    total_len: int = 128,
    prompt_lens: range | list[int] = range(16, 128),
    skip_first_source: int = 16,
    mask_id: int = MASK_ID,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Builds a corpus of "real context + one trailing [MASK]" examples,
    matching exactly what a single-mask eval-time readout (e.g.
    evaluation.ipynb's call_llada_lens) constructs at inference.

    For each prompt length ``p`` in ``prompt_lens``, truncates each input
    text to ``p`` tokens and appends a single ``mask_id`` token -- no
    sampling, no model forward pass. Each such example gets exactly one
    target position (the appended mask) and a source mask covering the real
    context (minus a fixed leading margin), mirroring
    :func:`build_diffusion_examples`'s source/target rationale but for a
    single, fixed noise level rather than a swept trajectory.

    Args:
        tokenizer: Tokenizer for ``prompts_text`` (the LLaDA tokenizer).
        prompts_text: Seed prompts to build contexts from.
        total_len: Fixed prompt+mask token budget; the prompt itself is
            truncated to at most ``total_len - 1`` tokens to always leave
            room for exactly one trailing mask token.
        prompt_lens: Prompt lengths to sweep, one example per
            ``(text, prompt_len)`` pair (not one per denoising step --
            there is no trajectory here, so this is far cheaper than the
            same range passed to :func:`build_diffusion_examples`).
        skip_first_source: Fixed leading margin excluded from the source
            mask, independent of ``prompt_len`` -- same rationale as in
            :func:`build_diffusion_examples`.
        mask_id: The model's mask-token id.

    Returns:
        ``(examples, source_masks, target_masks)`` -- no ``tau_bins``, since
        every example belongs to the same single class by construction.
        Texts shorter than ``total_len - 1`` tokens are skipped (not enough
        budget for the longest ``prompt_lens`` variant); ``prompt_lens``
        values at or beyond a given text's length are skipped for that text
        specifically, rather than truncating ``total_len`` inconsistently
        across the corpus. ``prompt_lens`` values ``<= skip_first_source``
        are skipped globally (not per-text): the sequence here is only
        ``prompt_len + 1`` tokens long with no response tail to fall back
        on, so if the whole prompt falls inside the skip_first_source
        margin, the source mask has nothing left --
        ``fitting.revealed_position_mask`` raises rather than returning an
        empty mask, so this is skipped before it can happen.
    """
    examples: list[torch.Tensor] = []
    source_masks: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []

    max_prompt_len = total_len - 1  # -1: always leave room for the trailing mask
    prompt_lens_list = list(prompt_lens)

    for text in prompts_text:
        full_ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_prompt_len
        ).input_ids
        if full_ids.shape[1] < max_prompt_len:
            continue

        for prompt_len in prompt_lens_list:
            # NEW: guard against the documented edge case -- if the entire
            # prompt falls within the skip_first_source margin, the source
            # mask has nothing left (unlike build_diffusion_examples, there
            # is no response tail here to fall back on), and
            # revealed_position_mask raises ValueError rather than silently
            # returning an empty mask. Skip these variants instead of
            # crashing the whole corpus build over one bad prompt_len.
            if prompt_len <= skip_first_source:
                continue

            prompt_ids = full_ids[:, :prompt_len]
            mask_col = torch.full((1, 1), mask_id, dtype=torch.long)
            augmented = torch.cat([prompt_ids, mask_col], dim=1)
            seq_len = augmented.shape[1]  # == prompt_len + 1

            # target: exactly the appended mask position (skip_first=prompt_len
            # leaves only the single trailing position True; exclude_last=False
            # keeps it -- there is no privileged "final position" convention
            # to exclude here, and that final position IS the one mask we want).
            tgt_mask = valid_position_mask(
                seq_len, skip_first=prompt_len, exclude_last=False
            )
            # source: real, never-masked context only (the mask position
            # itself is automatically excluded by revealed_position_mask's
            # != mask_id check), minus the fixed attention-sink margin.
            src_mask = revealed_position_mask(
                augmented, mask_id, skip_first=skip_first_source
            )
            examples.append(augmented)
            source_masks.append(src_mask)
            target_masks.append(tgt_mask)

    return examples, source_masks, target_masks


# NEW: teacher-forced diffusion trajectory -- for eval-time use in
# global_workspace_experiments.ipynb's "type B" experiments (directed-
# modulation, dual-task, selectivity-language, selectivity-linecount,
# top-down-summoning's Q1/Q2 readout, capacity).
#
# Rationale: these experiments read a J-lens over a SPAN of positions that
# hold real, already-given content (a teacher-forced carrier sentence the
# model is instructed to write, or a passage/question/list given to the
# model to read) -- never a position the lens was trained to treat as
# target=[MASK]. Appending a trailing [MASK] (the fix used for "type A",
# single-position experiments) does not apply here: it would either break
# the teacher-forcing (directed-modulation/dual-task -- the model must
# still produce the EXACT carrier text) or make no sense at all (there is
# no "next" position to predict for a given passage/question/list -- the
# whole point is reading positions WITHIN it).
#
# This function instead reconstructs the KNOWN target span (carrier
# sentence, question text, passage, or word list) via the same realistic,
# confidence-ordered reveal process as collect_trajectory -- but the
# revealed CONTENT is forced to the known target (teacher forcing) rather
# than the model's own argmax guess. Reading happens on each snapshot's
# STILL-MASKED positions, which is exactly the target role the J-lens
# bins were fit on.
#
# sequential_order=True forces strict left-to-right reveal instead of
# confidence-based reveal -- required for capacity specifically, where
# "how many of the first N words are still visible" is only meaningful if
# N really is the number of words read so far, in order.
@torch.no_grad()
def collect_teacher_forced_trajectory(
    hf_model,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    steps: int,
    mask_id: int = MASK_ID,
    device: str | torch.device = "cuda",  # CHANGED: caller sites (global_workspace_experiments.ipynb) pass a torch.device (e.g. prompt_ids.device), not just a string -- torch already accepted both at runtime, this just makes the annotation honest.
    sequential_order: bool = False,
) -> list[dict]:
    """Runs a teacher-forced diffusion reveal of ``target_ids`` after
    ``prompt_ids``, using the model's own confidence to choose reveal order
    (matching real inference) while forcing revealed content to the known
    target (matching Llama's teacher forcing for the same experiments).

    Args:
        hf_model: Raw HF LLaDA model (needs plain ``.logits`` output).
        prompt_ids: Tensor of shape ``[1, prompt_len]`` -- real, never-masked
            context preceding the span to reconstruct.
        target_ids: Tensor of shape ``[1, target_len]`` -- the known content
            to reveal (carrier sentence / question / passage / word list,
            already tokenized).
        steps: Number of reveal steps. ``steps == target_len`` gives one
            token revealed per step (matches the paper's default regime,
            see the diffusion notebooks' rationale for pure diffusion).
        mask_id: The model's mask-token id.
        device: Device to run on.
        sequential_order: If ``True``, reveal strictly left-to-right instead
            of by confidence. Needed only for capacity (see module-level
            rationale above); leave ``False`` elsewhere for a realistic
            reveal order.

    Returns:
        A list of ``{"input_ids", "frac_masked", "still_masked_positions"}``
        dicts, one per step, in trajectory order. Snapshots are taken
        BEFORE each step's forcing, so ``still_masked_positions`` always
        includes the position(s) about to be revealed -- these are exactly
        the target-role positions the J-lens bins were fit to read.
    """
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]
    x = torch.full(
        (1, prompt_len + target_len), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt_ids.to(device)
    target_full = torch.cat([prompt_ids.to(device), target_ids.to(device)], dim=1)

    snapshots: list[dict] = []
    num_transfer = _num_transfer_tokens(x[:, prompt_len:] == mask_id, steps)

    for i in range(steps):
        mask_index = x == mask_id
        if mask_index.sum() == 0:
            break

        still_masked_positions = mask_index[0].nonzero(as_tuple=True)[0].tolist()
        snapshots.append(
            {
                "input_ids": x.clone(),
                "frac_masked": (x[:, prompt_len:] == mask_id).float().mean().item(),
                "still_masked_positions": still_masked_positions,
            }
        )

        k = int(num_transfer[0, i].item())
        if k == 0:
            continue

        if sequential_order:
            # Strictly left-to-right: reveal the k lowest-index masked
            # positions, regardless of model confidence.
            idx = torch.tensor(still_masked_positions[:k], device=x.device)
        else:
            # Realistic order: same confidence computation as
            # collect_trajectory, but the CONTENT committed is target_full
            # (teacher forcing), not the model's own argmax.
            logits = hf_model(x).logits
            p = torch.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(logits, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=x.device)
            )
            _, idx = torch.topk(confidence[0], k=k)

        x[0, idx] = target_full[0, idx]

    return snapshots


# NEW: shared readout helper for the same 5 "type B" experiments -- runs
# collect_teacher_forced_trajectory once, reads hit_rate/best_rank_over_band
# on each snapshot's still-masked positions with THAT snapshot's tau-bin
# lens, and aggregates per bin. Kept here (not in the experiments notebook)
# so both fitting notebooks and the experiments notebook share one
# definition of "which bin a given frac_masked belongs to" -- must match
# build_diffusion_examples' own tau_bins formula exactly, or a bin-lens
# would be applied to snapshots outside the noise-level range it was fit on.
def bin_for_frac_masked(frac_masked: float, n_bins: int = 4) -> int:
    """Same formula as build_diffusion_examples' tau_bins.append(...) line --
    kept as a standalone function so eval-time code and fitting-time code
    can never drift apart on this."""
    return min(int(frac_masked * n_bins), n_bins - 1)