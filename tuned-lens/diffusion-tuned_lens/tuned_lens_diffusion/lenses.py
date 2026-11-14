"""The diffusion tuned lens.

A per-layer residual translator ``h + W_l h + b_l`` (zero-init, so training
starts at the identity = logit lens) followed by the model's frozen unembed.
This is the tuned lens of Belrose et al., unchanged; the diffusion-specific
part lives in the corpus (which snapshots and which target positions) and the
training loop, not here.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch as th

from .model import LLaDAUnembed, model_dims


@dataclass
class DiffusionTunedLensConfig:
    """Everything needed to rebuild a lens, incl. diffusion bookkeeping."""

    base_model_name_or_path: str
    d_model: int
    num_hidden_layers: int
    bias: bool = True
    n_prompts: int = 0
    """Number of snapshots the lens was fit on (used to weight merge())."""
    tau_bin: int | None = None
    """Which diffusion-stage bin this lens was fit on (None once merged)."""
    target_mode: str = "mask"
    """Which positions the translators were trained on (currently only "mask")."""


class DiffusionTunedLens(th.nn.Module):
    """Residual tuned lens for a masked-diffusion LM (LLaDA)."""

    def __init__(self, unembed: LLaDAUnembed, config: DiffusionTunedLensConfig):
        super().__init__()
        self.unembed = unembed
        self.config = config

        # fp32 translators (training is done in fp32 even when the base model
        # is bf16); zero-init => identity at the start, i.e. a logit lens.
        translator = th.nn.Linear(config.d_model, config.d_model, bias=config.bias, dtype=th.float32)
        th.nn.init.zeros_(translator.weight)
        if translator.bias is not None:
            th.nn.init.zeros_(translator.bias)
        self.layer_translators = th.nn.ModuleList(
            [deepcopy(translator) for _ in range(config.num_hidden_layers)]
        )

    def __getitem__(self, idx: int) -> th.nn.Module:
        return self.layer_translators[idx]

    def __iter__(self):
        yield from self.layer_translators

    def __len__(self) -> int:
        return len(self.layer_translators)

    def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
        """``h + W_l h + b_l`` in fp32 (residual parametrization of the
        upstream tuned lens: weight decay pulls toward identity, not zero)."""
        x = h.float()
        return x + self[idx](x)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        """Translated residual -> logits via the frozen unembed."""
        return self.unembed(self.transform_hidden(h, idx))

    # ------------------------------------------------------------------ #
    # construction / persistence
    # ------------------------------------------------------------------ #
    @classmethod
    def from_model(
        cls,
        hf_model: th.nn.Module,
        *,
        bias: bool = True,
        target_mode: str = "mask",
        tau_bin: int | None = None,
    ) -> "DiffusionTunedLens":
        n_layers, d_model = model_dims(hf_model)
        config = DiffusionTunedLensConfig(
            base_model_name_or_path=getattr(hf_model.config, "name_or_path", "") or "",
            d_model=d_model,
            num_hidden_layers=n_layers,
            bias=bias,
            target_mode=target_mode,
            tau_bin=tau_bin,
        )
        return cls(LLaDAUnembed(hf_model), config)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        th.save(self.layer_translators.state_dict(), path / "params.pt")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, path: str | Path, *, hf_model: th.nn.Module) -> "DiffusionTunedLens":
        path = Path(path)
        config = DiffusionTunedLensConfig(**json.loads((path / "config.json").read_text()))
        lens = cls(LLaDAUnembed(hf_model), config)
        lens.layer_translators.load_state_dict(
            th.load(path / "params.pt", map_location="cpu", weights_only=True)
        )
        return lens

    @classmethod
    def merge(cls, lenses: "list[DiffusionTunedLens]") -> "DiffusionTunedLens":
        """Weighted (by ``n_prompts``) merge of per-bin lenses into one, the
        tuned-lens analogue of ``jlens.lens.JacobianLens.merge``."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.config.d_model != first.config.d_model
                or other.config.num_hidden_layers != first.config.num_hidden_layers
            ):
                raise ValueError("lenses disagree on d_model / num_hidden_layers")
        total = sum(lens.config.n_prompts for lens in lenses)
        if total == 0:
            weights = [1.0 / len(lenses)] * len(lenses)
        else:
            weights = [lens.config.n_prompts / total for lens in lenses]

        config = DiffusionTunedLensConfig(
            base_model_name_or_path=first.config.base_model_name_or_path,
            d_model=first.config.d_model,
            num_hidden_layers=first.config.num_hidden_layers,
            bias=first.config.bias,
            n_prompts=total,
            tau_bin=None,
            target_mode=first.config.target_mode,
        )
        merged = cls(first.unembed, config).to(first[0].weight.device)
        with th.no_grad():
            for i in range(config.num_hidden_layers):
                w = sum(lens[i].weight.data * weight for lens, weight in zip(lenses, weights))
                merged[i].weight.data.copy_(w)
                if config.bias:
                    b = sum(lens[i].bias.data * weight for lens, weight in zip(lenses, weights))
                    merged[i].bias.data.copy_(b)
        return merged
