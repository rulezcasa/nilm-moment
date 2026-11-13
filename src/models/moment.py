"""MOMENT (Goswami et al., ICML 2024) adapted to sequence-to-point.

The backbone is momentfm's MOMENTPipeline: 8-point patches and a T5 encoder.  The head is the paper's flatten + linear forecasting head with
horizon 1.  RevIN is disabled: the encoder sees the dataset-level normalized
input like the other models.
"""

import json
from typing import Any, Mapping

from torch import Tensor, nn


class _NoRevIN(nn.Module):
    """Pass-through in place of momentfm's RevIN."""

    def forward(self, x: Tensor, mask: Tensor = None, mode: str = "norm") -> Tensor:
        return x


class MOMENTNILM(nn.Module):
    """Predict the target appliance at the midpoint from MOMENT patch embeddings."""

    def __init__(
        self,
        input_length: int,
        checkpoint: str,
        pretrained: bool,
        freeze_encoder: bool,
        head_dropout: float,
    ) -> None:
        super().__init__()
        # Imported here so the other models run without momentfm installed.
        from huggingface_hub import hf_hub_download
        from momentfm import MOMENTPipeline

        model_kwargs = {
            "task_name": "embedding",
            "freeze_embedder": freeze_encoder,
            "freeze_encoder": freeze_encoder,
            "enable_gradient_checkpointing": False,
        }
        if pretrained:
            self.backbone = MOMENTPipeline.from_pretrained(checkpoint, model_kwargs=model_kwargs)
        else:
            # Same architecture from the checkpoint's config, without its weights.
            with open(hf_hub_download(checkpoint, "config.json")) as handle:
                self.backbone = MOMENTPipeline(json.load(handle), model_kwargs=model_kwargs)
        self.backbone.init()
        if input_length != self.backbone.seq_len:
            raise ValueError(f"MOMENT expects input_length {self.backbone.seq_len}, got {input_length}")

        self.backbone.normalizer = _NoRevIN()
        patches = input_length // self.backbone.patch_len
        self.dropout = nn.Dropout(head_dropout)
        self.regressor = nn.Linear(patches * self.backbone.config.d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.backbone.embed(x_enc=x.unsqueeze(1), reduction="none").embeddings
        return self.regressor(self.dropout(embeddings.flatten(1)))


def build_moment(model_cfg: Mapping[str, Any], input_length: int) -> MOMENTNILM:
    """Construct MOMENT from a config's ``model`` section."""

    return MOMENTNILM(input_length, **{key: value for key, value in model_cfg.items() if key != "name"})
