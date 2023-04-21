# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation script."""

import functools
from os import path
import sys
import time

from absl import app
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import raw_utils
from internal import ref_utils
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):
  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('all', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)
  cc_fun = image.color_correct

  last_step = 0

  # --
  out_dir = path.join(config.checkpoint_dir, 'results')
  path_fn = lambda x: path.join(out_dir, x)

  output_path = Path(out_dir)
  output_path.mkdir(exist_ok=True, parents=True)
  (output_path / 'images').mkdir(exist_ok=True, parents=True)
  (output_path / 'depths').mkdir(exist_ok=True, parents=True)

  data_path = Path(config.data_dir)
  shutil.copy(data_path / 'transforms.json', output_path / 'transforms.json')
  # --


  while True:
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    step = int(state.step)
    if step <= last_step:
      print(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
      time.sleep(10)
      continue
    print(f'Evaluating checkpoint at step {step}.')
    break
  key = random.PRNGKey(0 if config.deterministic_showcase else step)
  perm = random.permutation(key, dataset.size)

  for idx in tqdm(range(dataset.size), total=dataset.size, dynamic_ncols=True):
    batch = next(dataset)
    rendering = models.render_image(
        functools.partial(render_eval_pfn, state.optimizer.target),
        batch['rays'],
        None,
        config)

    utils.save_img_u8(rendering['rgb'], output_path / 'images' / f'{idx:06d}.rgb.png')
    utils.save_img_f32(rendering['distance_mean'], output_path / 'depths' / f'{idx:06d}.depth.tiff')
    utils.save_img_f32(rendering['distance_median'], output_path / 'depths' / f'{idx:06d}.depthm.tiff')

if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
