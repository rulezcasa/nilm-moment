"""BERT4NILM (Yue et al., NILM 2020) adapted to sequence-to-point.

Layers match the reference implementation; the midpoint of the decoded
sequence is returned and training uses this repo's MSE objective, not the
paper's masked multi-term loss.
"""

from typing import Any, Mapping

from torch import Tensor, nn

from src.data.windowing import anchor_offset


class TransformerBlock(nn.Module):
    """Post-LN encoder block with dropout after the block, as in the reference."""

    def __init__(self, hidden_size: int, attention_heads: int, feed_forward_size: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, attention_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feed_forward_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(feed_forward_size, hidden_size),
        )
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.feed_forward_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention_norm(x + self.dropout(self.attention(x, x, x, need_weights=False)[0]))
        x = self.feed_forward_norm(x + self.dropout(self.feed_forward(x)))
        return self.dropout(x)


class BERT4NILM(nn.Module):
    """Predict the target appliance at the midpoint of an aggregate window."""

    def __init__(
        self,
        input_length: int,
        hidden_size: int,
        attention_heads: int,
        transformer_layers: int,
        feed_forward_size: int,
        dropout: float,
        conv_kernel_size: int = 5,
        downsample_factor: int = 2,
        decoder_kernel_size: int = 4,
        decoder_stride: int = 2,
        dense_units: int = 128,
    ) -> None:
        super().__init__()
        if input_length % downsample_factor != 0:
            raise ValueError("input_length must be divisible by downsample_factor")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")

        self.target_index = anchor_offset(input_length)
        latent_length = input_length // downsample_factor
        conv_padding = conv_kernel_size // 2

        self.embedding = nn.Conv1d(
            1,
            hidden_size,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
            padding_mode="replicate",
        )
        self.pool = nn.LPPool1d(2, kernel_size=downsample_factor, stride=downsample_factor)
        self.position = nn.Embedding(latent_length, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.input_dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(
            *(
                TransformerBlock(hidden_size, attention_heads, feed_forward_size, dropout)
                for _ in range(transformer_layers)
            )
        )
        self.decoder = nn.ConvTranspose1d(
            hidden_size,
            hidden_size,
            kernel_size=decoder_kernel_size,
            stride=decoder_stride,
            padding=(decoder_kernel_size - decoder_stride) // 2,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, dense_units),
            nn.Tanh(),
            nn.Linear(dense_units, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Truncated-normal init on every non-LayerNorm parameter, as in the reference."""

        for name, parameter in self.named_parameters():
            if "norm" not in name:
                nn.init.trunc_normal_(parameter, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.pool(self.embedding(x.unsqueeze(1))).transpose(1, 2)
        positions = self.position.weight.unsqueeze(0).expand(tokens.size(0), -1, -1)
        tokens = self.input_dropout(self.input_norm(tokens + positions))
        tokens = self.transformer(tokens)
        decoded = self.decoder(tokens.transpose(1, 2)).transpose(1, 2)
        return self.regressor(decoded[:, self.target_index, :])


def build_bert4nilm(model_cfg: Mapping[str, Any], input_length: int) -> BERT4NILM:
    """Construct BERT4NILM from a config's ``model`` section."""

    return BERT4NILM(input_length, **{key: value for key, value in model_cfg.items() if key != "name"})
