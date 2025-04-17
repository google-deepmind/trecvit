# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TrecViT model."""

from typing import Callable, Optional

import einops
import flax.linen as nn
from flax.typing import Array  # pylint: disable=g-importing-member,g-multiple-import
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import modules as ssm_modules

from trecvit import utils


def transpose_flatten(
    x: jnp.ndarray, like_shape: tuple[int, int, int, int], original_shape: str
) -> jnp.ndarray:
  shape = dict(zip("btnc", like_shape, strict=True))
  return einops.rearrange(x, original_shape + "-> (b n) t c", **shape)


def unflatten_untranspose(
    x: jnp.ndarray, like_shape: tuple[int, int, int, int], original_shape: str
) -> jnp.ndarray:
  shape = dict(zip("btnc", like_shape, strict=True))
  return einops.rearrange(x, "(b n) t c ->" + original_shape, **shape)


class Tokenizer(nn.Module):
  """Video tokenizer.

  Takes a video ([b, t, h, w, c]) and outputs tokens ([b, t, n_patches, width]).
  """

  width: int = 768
  patch_size: tuple[int, int, int] = (1, 16, 16)
  posemb: str = "learn"
  dtype: str = "float32"
  pool_type: str = "tok"

  @nn.compact
  def __call__(self, x: jnp.ndarray, override_drop_ratio: float | None = None):
    out = {}
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding="VALID",
        name="embedding",
        dtype=self.dtype,
        param_dtype=self.dtype,
    )(x)
    b, t, h, w, c = x.shape
    out["patch_shape"] = x.shape

    x = jnp.reshape(x, [b * t, h * w, c])
    x = x + utils.get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype
    )

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [x.shape[0], 1, 1]), x], axis=1)

    x = jnp.reshape(x, [b, t, -1, c])
    out["tokens"] = x
    return out


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Forked from:
  https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py#L57
  """

  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    d = x.shape[-1]
    x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
    # In some extreme batch-size cases, this is needed as of Sept 2024:
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP).

  Forked from:
  https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py#L81
  """

  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  attention_fn: Callable[..., Array] = nn.dot_product_attention
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype_mm,
        attention_fn=self.attention_fn,
    )(y, y)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x, out


class LRUViT(nn.Module):
  """Video Encoder with Spatial Attention and Temporal LRU (SSM)."""

  depth: int
  width: int
  mlp_dim: int | None = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  lru_width: int | None = None
  conv1d_temporal_width: int = 2
  attention_window_size: int = 2048
  scan_type: common.ScanType = common.ScanType.AUTO
  dtype: str = "float32"
  only_real: bool = True
  min_rad: float = 0.7
  use_mlp: bool = False
  state_multiplier: int = 1
  is_training: bool = False

  @nn.compact
  def __call__(self, x):
    b, t, n, c = x.shape
    pos = jnp.repeat(jnp.arange(t)[None], b * n, axis=0)
    x = jnp.reshape(x, [b * t, n, c])
    temporal_block_types = [common.TemporalBlockType.RECURRENT] * self.depth
    temporal_layers = tuple(
        ssm_modules.ResidualBlock(
            name=f"residualblock_{i:02d}",
            width=self.width,
            mlp_expanded_width=self.mlp_dim,
            num_heads=self.num_heads,
            lru_width=self.state_multiplier * self.width,
            conv1d_temporal_width=self.conv1d_temporal_width,
            attention_window_size=self.attention_window_size,
            temporal_block_type=temporal_block_type,
            scan_type=self.scan_type,
            final_w_init_variance_scale=2.0 / self.depth,
            dtype=self.dtype,
            param_dtype=self.dtype,
            lru_only_real=self.only_real,
            min_rad=self.min_rad,
            use_mlp=self.use_mlp,
        )
        for i, temporal_block_type in enumerate(temporal_block_types)
    )

    # Input Encoder
    for lyr in range(self.depth):
      x = transpose_flatten(x, (b, t, n, c), "(b t) n c")
      x, _ = temporal_layers[lyr](x, pos)
      x = unflatten_untranspose(x, (b, t, n, c), "(b t) n c")
      block_cur = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          dtype_mm=self.dtype,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
      )
      x, _ = block_cur(x, not self.is_training)
    x = jnp.reshape(x, [b, t, n, c])  # output has the same shape as input
    x = nn.LayerNorm(name="encoder_norm")(x)

    return x


class ClassificationDecoder(nn.Module):
  """Video Decoder."""

  num_classes: int
  width: int = 768
  rep_size: int = 3072
  patch_size: tuple[int, int, int] = (1, 16, 16)
  aggregation: str = "temporal_avg_pooling"
  dropout: float = 0.0
  is_training: bool = False
  pool_type: str = "tok"
  dtype: str = "float32"

  @nn.compact
  def __call__(self, x, b, t, num_clips):
    out = dict()
    x = jnp.reshape(x, [b * t, -1, self.width])

    if self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0, :]
    else:
      raise ValueError(f"Unknown pooling type: {self.pool_type}")

    hid = nn.Dense(self.rep_size, name="pre_logits")
    x = nn.tanh(hid(x))
    t_out = t // (num_clips * self.patch_size[0])
    x = einops.rearrange(
        x,
        "(b n t) d -> b n t d",
        t=t_out,
        n=num_clips,
    )
    # Temporal pooling
    pre_logits = einops.reduce(x, "b n t d -> b n d", reduction="mean")
    pre_logits = nn.Dropout(rate=self.dropout)(pre_logits, not self.is_training)

    logits = nn.Dense(self.num_classes, name="cls_head")(pre_logits)
    probs = nn.activation.softmax(logits, axis=-1)

    # Pooling the output of multiple video clips sampled from the same video.
    logits = einops.reduce(logits, "b n d -> b d", reduction="mean")
    probs = einops.reduce(probs, "b n d -> b d", reduction="mean")

    out["pre_logits"] = pre_logits
    out["logits"] = logits
    out["probs"] = probs

    return out


class TRecViT(nn.Module):
  """Wrapper for running the LRUViT encoder on video sequences.

  Works in three stages:

  1. Tokenize the video.
      input is [b, t, h, w, c]
      output is [b, t, n_patches, width]

  2. Encode the tokens.
     input is [b, t, n_patches, width]
     output is [b, t, n_patches, width]

  3. Decode the tokens.
    input is [b, num_clips, t, width]
    output is [b, num_classes]
  """

  tokenizer: nn.Module
  encoder: nn.Module
  decoder: nn.Module
  frames_per_clip: int = 16

  @nn.compact
  def __call__(
      self, video: jnp.ndarray, override_drop_ratio: float | None = None
  ):
    b, t, _, _, _ = video.shape
    out = dict()
    num_clips = max(int(t // self.frames_per_clip), 1)
    video = einops.rearrange(
        video, "... (n t) h w c -> (... n) t h w c", n=num_clips
    )

    tmp = self.tokenizer(video, override_drop_ratio=override_drop_ratio)
    out.update(tmp)
    encoded = self.encoder(out["tokens"])
    out["encoded"] = encoded
    tmp = self.decoder(encoded, b, t, num_clips)
    out.update(tmp)
    return out
