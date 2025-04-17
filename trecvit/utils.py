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

"""Utilities for loading checkpoints.

These functions are used to load checkpoints.
Most of these functions are forked from
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""

import collections
from typing import Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1.0 / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(
        name,
        nn.initializers.normal(stddev=1 / np.sqrt(width)),
        (1, np.prod(seqshape), width),
        dtype,
    )
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


def decode_variant(variant: str) -> dict[str, Any]:
  """Converts a string like "B" or "B/32" into a params dict.

  Args:
    variant: A string like "B" or "B/32".

  Returns:
    A dictionary of parameters.
  """
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {
          "mu": 32,
          "Ti": 192,
          "S": 384,
          "M": 512,
          "B": 768,
          "L": 1024,
          "So400m": 1152,
          "H": 1280,
          "g": 1408,
          "g-opt": 1536,
          "G": 1664,
          "G-opt": 1536,
          "e": 1792,
      }[v],
      "depth": {
          "mu": 1,
          "Ti": 12,
          "S": 12,
          "M": 12,
          "B": 12,
          "L": 24,
          "So400m": 27,
          "H": 32,
          "g": 40,
          "g-opt": 40,
          "G": 48,
          "G-opt": 48,
          "e": 56,
      }[v],
      "mlp_dim": {
          "mu": 128,
          "Ti": 768,
          "S": 1536,
          "M": 2048,
          "B": 3072,
          "L": 4096,
          "So400m": 4304,
          "H": 5120,
          "g": 6144,
          "g-opt": 6144,
          "G": 8192,
          "G-opt": 8192,
          "e": 15360,
      }[v],
      "num_heads": {
          "mu": 2,
          "Ti": 3,
          "S": 6,
          "M": 8,
          "B": 12,
          "L": 16,
          "So400m": 16,
          "H": 16,
          "g": 16,
          "g-opt": 16,
          "G": 16,
          "G-opt": 16,
          "e": 16,
      }[v],
      # pylint:enable=line-too-long
      **patch,
  }


def _recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.

  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = _recover_tree(k_subtree, v_subtree)
  return tree


def load_ckpt(
    init_params: dict[str, Any],
    restored_params_path: dict[str, Any],
    dtype: jnp.dtype = jnp.float32,
) -> dict[str, Any]:
  """Transform the state by updating it with pre-trained weights.

  Args:
    init_params: Initial parameters to be updated.
    restored_params_path: The path to the restored parameters.
    dtype: The dtype to use for the parameters.

  Returns:
    The updated parameters.
  """
  restored_params_raw = dict(np.load(restored_params_path, allow_pickle=False))
  keys, values = zip(*list(restored_params_raw.items()))
  restored_params = _recover_tree(keys, values)

  for block_key in ["decoder", "encoder", "tokenizer"]:
    for module_name, module_params in restored_params[block_key].items():
      restored_params_flat, _ = jax.tree.flatten_with_path(module_params)
      train_params_flat, train_params_leaves = jax.tree.flatten_with_path(
          init_params["params"][block_key][module_name]
      )
      if len(restored_params_flat) != len(train_params_flat):
        raise ValueError(
            f"""Number of named parameters in restored checkpoint:
            ({len(restored_params_flat)}) does not match number of named
            parameters in initial checkpoint: ({len(train_params_flat)}).""",
        )

      new_params = []
      for rp, tp in zip(restored_params_flat, train_params_flat):
        restored_params_k = "".join([x.key + "/" for x in rp[0]])[:-1]
        train_params_k = "".join([x.key + "/" for x in tp[0]])[:-1]

        if restored_params_k == train_params_k and rp[1].shape == tp[1].shape:
          new_params.append(rp[1].astype(dtype))
        else:
          raise ValueError("Wrong shape: ", rp[1].shape, tp[1].shape)

      unflattened_block = jax.tree.unflatten(train_params_leaves, new_params)
      init_params["params"][block_key][module_name] = unflattened_block

  return init_params
