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

"""TRecViT model.

TRecViT is a causal recurrent video transformer model that can process
videos in real time and perform various tasks like action recognition,
point tracking.

See paper for more details: https://arxiv.org/pdf/2412.14294
"""

from trecvit import modules
from trecvit import utils


def get_model(
    model_size: str = "B", num_frames: int = 32, num_classes: int = 400
) -> modules.TRecViT:
  """Returns a TrecViT model.

  Args:
    model_size: The size of the model to use.
    num_frames: The number of frames to use. Default is 32 for Kinetics400.
    num_classes: The number of classes to use. Default is 400 for Kinetics400.

  Returns:
    The TrecViT model.
  """
  model_info = utils.decode_variant(model_size)
  width = model_info["width"]
  depth = model_info["depth"]
  mlp_dim = model_info["mlp_dim"]
  num_heads = model_info["num_heads"]
  patch_size = (1, 16, 16)

  aux = {
      "conv1d_temporal_width": 4,
      "state_multiplier": 2,
      "posemb": "learn",
      "pool_type": "tok",
      "use_mlp": False,
      "rep_size": 3072,
      "min_rad": 0.5,
  }

  tokenizer = modules.Tokenizer(
      width=width,
      patch_size=patch_size,
      posemb=aux["posemb"],
      pool_type=aux["pool_type"],
  )

  encoder = modules.LRUViT(
      depth=depth,
      width=width,
      mlp_dim=mlp_dim,
      num_heads=num_heads,
      only_real=True,
      conv1d_temporal_width=aux["conv1d_temporal_width"],
      state_multiplier=aux["state_multiplier"],
      use_mlp=aux["use_mlp"],
      min_rad=aux["min_rad"],
  )

  decoder = modules.ClassificationDecoder(
      num_classes=num_classes,
      width=width,
      rep_size=aux["rep_size"],
      pool_type=aux["pool_type"],
  )

  model = modules.TRecViT(
      tokenizer=tokenizer,
      encoder=encoder,
      decoder=decoder,
      frames_per_clip=num_frames,
  )
  return model
