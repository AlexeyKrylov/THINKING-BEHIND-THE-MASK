# Thinking Behind the Mask: A Global Workspace in Diffusion Language Models

Code for the paper *Thinking Behind the Mask: A Global Workspace in Diffusion Language Models*
(A. S. Krylov, I. R. Ashirov, O. Y. Rogov, 2026).

The Jacobian lens (J-lens) of [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html)
decodes an intermediate residual stream with the corpus-averaged Jacobian of the final layer
and reveals a "global workspace" of verbalizable representations in autoregressive LLMs.
This repository adapts the J-lens, the logit lens and the tuned lens to masked diffusion
language models and compares **LLaDA-8B-Instruct** with **Llama-3.1-8B-Instruct** on six
lens-quality suites and eleven global-workspace experiments.

The main LLaDA lens (*merged*) averages the Jacobian over the model's own denoising
trajectories: sources are already revealed positions, targets are all response positions,
masked ones included, and the trajectory snapshots are split into four noise-level bins whose
lenses are merged into one.

Lens weights are not stored in the repository (each J-lens is about 2 GB); the notebooks below
rebuild all of them.

## Repository layout

| Path | Contents |
|---|---|
| `fit_lens_0.ipynb` | J-lens of Llama-3.1-8B-Instruct on clean WikiText (and an unused clean-text LLaDA lens) |
| `fit_lens_diffusion.ipynb` | Diffusion J-lenses of LLaDA: trajectory corpus, four bin lenses and the merged lens |
| `fit_lens_single_mask_1.ipynb` | Auxiliary single-mask J-lens of LLaDA ("context + one `[MASK]`") |
| `tuned-lens/fit_tuned_lens_llama.ipynb` | Tuned lens of Llama-3.1-8B-Instruct |
| `tuned-lens/fit_tuned_lens_llada.ipynb` | Diffusion tuned lens of LLaDA-8B-Instruct |
| `evaluation_baselines.ipynb` | Lens-quality evaluation: J-lens vs tuned lens vs logit lens, pass@k and AUC |
| `global_workspace_experiments-parallel.ipynb` | The eleven global-workspace experiments for all lenses, multi-GPU |
| `jacobian-lens/` | [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) with the diffusion extensions, evaluation suites and experiment datasets |
| `tuned-lens/ar-tuned_lens-repo/` | The `tuned_lens` package ([tuned-lens](https://github.com/AlignmentResearch/tuned-lens)) |
| `tuned-lens/diffusion-tuned_lens/` | `tuned_lens_diffusion`: tuned lens for masked diffusion models |

Changes to `jlens` relative to the upstream repository:

- `jlens/diffusion_corpus.py` (new): the LLaDA sampler that records a snapshot after every
  denoising step (`collect_trajectory`), the trajectory corpus with source/target masks and
  noise-level bins (`build_diffusion_examples`), the single-mask corpus
  (`build_single_mask_examples`) and helpers for the `[MASK]` read-out.
- `jlens/fitting.py`: `fit` accepts separate source and target position masks per example;
  `valid_position_mask` gets `exclude_last`, since a diffusion model has no privileged last position.
- `data/evaluations/lens-eval-poetry.json`: prompts are cut so that the last position is the
  read-out position (the end of the first line of the couplet).

## Installation

```bash
git clone <this repository>
cd thinking-behind-the-mask
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./jacobian-lens
```

`tuned_lens` and `tuned_lens_diffusion` are imported from `tuned-lens/` by the notebooks
themselves (`sys.path`), they do not need to be installed.

Llama-3.1-8B-Instruct is a gated model. Request access on Hugging Face and export a token
before starting Jupyter:

```bash
export HF_TOKEN=<your token>
```

All notebooks are run from the repository root, except the two tuned-lens notebooks, which
are run from `tuned-lens/`. The experiments in the paper were run in bfloat16 on NVIDIA
H100 80GB GPUs. Every notebook uses `cuda:0` by default; change `device` in its first cells
to use another GPU.

## Reproducing the paper

### 1. Lenses

| Step | Notebook | Output (repository root unless noted) |
|---|---|---|
| Llama J-lens | `fit_lens_0.ipynb`, section *Llama-3.1-8B-Instruct* | `llama_jacobian_lens_1000.pt` |
| LLaDA diffusion J-lenses | `fit_lens_diffusion.ipynb` | `llada_diffusion_corpus.pkl`, `new_llada_diffusion_lens1000_bin{0..3}.pt`, `new_llada_diffusion_lens_merged_1000.pt` |
| LLaDA single-mask J-lens | `fit_lens_single_mask_1.ipynb` | `llada_single_mask_lens_1000.pt` |
| Llama tuned lens | `tuned-lens/fit_tuned_lens_llama.ipynb` | `tuned-lens/ar-tuned_lens-repo/train_lenses/meta-llama/Llama-3.1-8B-Instruct/` |
| LLaDA tuned lens | `tuned-lens/fit_tuned_lens_llada.ipynb` | `tuned-lens/diffusion-tuned_lens/train_lenses/GSAI-ML/LLaDA-8B-Instruct/` |

Settings used in the paper:

- **Llama J-lens**: 1000 WikiText prompts, up to 128 tokens, the first 16 positions and the
  last one excluded (`jlens.fit` defaults). About 7 hours on one H100.
- **LLaDA diffusion J-lens**: 1000 WikiText texts of at least 128 tokens; prompt lengths
  16, 24, …, 120; the remaining `128 − p` tokens are generated by the LLaDA sampler in a single
  block, one token per step, with greedy low-confidence remasking (822,360 snapshots in total).
  Sources are revealed positions after the first 16, targets are all response positions.
  Snapshots are split into four bins by the masked fraction of the response and subsampled to
  2,500 per bin. The corpus is cached in `llada_diffusion_corpus.pkl`, so that the fitting can
  be restarted without resampling.
- **LLaDA single-mask J-lens**: every WikiText text is cut at context lengths 16–127 and
  followed by one `[MASK]`; 10,000 random examples; the target is the mask position.
- **Llama tuned lens**: 1000 WikiText prompts, up to 128 tokens, 50 SGD steps of 4096 tokens.
- **LLaDA tuned lens**: denoising trajectories of 50 WikiText texts (total length 128, two random
  prompt lengths from 16, 32, …, 112 per text), four noise-level bins with 250 snapshots each;
  per-layer translators are trained in one SGD pass (lr 1.0, momentum 0.9, weight decay 1e-3) on
  the KL divergence to the final distribution at masked positions, and the bin lenses are merged.

Long fits write checkpoints (`*_ckpt*.pt`) and resume from them after an interruption.

### 2. Lens-quality evaluation (Table 1, single-token table and pass@k figures)

`evaluation_baselines.ipynb` reads all three lenses on the six suites of
`jacobian-lens/data/evaluations/` over layers 0–30. For Llama the lens is read at the last
prompt position, for LLaDA at one `[MASK]` appended to the prompt. It reports pass@k for
k ∈ {1, 5, 20, 100, 200, 500, 1000} and the normalized AUC in two modes: *first-token* (the first
token of every intermediate word) and *single-token* (only the 305 words that are single tokens
for both tokenizers). The first cell finds the lens files and the tuned-lens code automatically
and prints the paths it uses.

### 3. Global-workspace experiments (Table 2 and the appendix tables)

`global_workspace_experiments-parallel.ipynb` runs the eleven experiments of
`jacobian-lens/data/experiments/` (verbal report, verbal introspection, directed modulation,
top-down summoning, probe-swap, flexible generalization, selectivity-language,
selectivity-linecount, ignition, capacity, dual-task) for the Llama J-lens, the six LLaDA
J-lenses (bins 0–3, merged, single-mask) and the logit and tuned lenses of both models.
Interventions and read-outs use the workspace band of layers 8–27.

Settings in the configuration cell:

- `GPU_IDS`: GPUs to use, e.g. `[0]` or `[0, 1, 2]`. Every GPU holds a copy of both models and
  needs about 40 GB of free memory; independent trials of every experiment are distributed over
  the GPUs. The paper used three H100.
- `DEBUG_MODE = True`: a fast run on a subset of items (capacity on every 10th comma, 20 ignition
  pairs instead of 66).
- `SWAP_ABLATION_MODES = ("transfer",)`: the swap ablation of the appendix; `()` disables it.
- `BASELINE_LENSES = ("logit-lens", "tuned-lens")`: baselines; `()` disables them.

Every experiment writes its own CSV and figure; the last cells collect all experiments into one
summary table and a heat map.