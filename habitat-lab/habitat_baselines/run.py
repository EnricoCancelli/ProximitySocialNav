#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random

import numpy as np
import torch

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config

import wandb
import yaml

from typing import Union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "collect", "votrain"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="name of wandb run",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--wandb_run",
        action='store_true'
    )
    parser.add_argument(
        "--seed",
        type=int
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Union[Config, dict], run_type: str, wandb_run=None, name="default") -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    if isinstance(config, Config):
        random.seed(config.P_SEED)  # TASK_CONFIG.SEED)
        np.random.seed(config.P_SEED)  # TASK_CONFIG.SEED)
        torch.manual_seed(config.P_SEED)  # TASK_CONFIG.SEED)
        if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
            torch.set_num_threads(1)

        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    else:
        trainer_init = baseline_registry.get_trainer(config["trainer"])
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

    wandb_ctx = None
    if wandb_run is not None and wandb_run:
        wandb_ctx = wandb.init(name=name,
                               project="**"
                               entity="**", config=dict(config))
    trainer = trainer_init(config, runtype=run_type, wandb_ctx=wandb_ctx)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "collect":
        trainer.collect_dataset()
    elif run_type == "votrain":
        trainer.train_vo()


PATHS_TO_JUNK = {}


def run_exp(exp_config: str, run_type: str, opts=None, wandb_run=None,
            seed=None, name="default") -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    if 'JUNK' in opts and opts[opts.index('JUNK') + 1] == 'True':
        for k, v in PATHS_TO_JUNK.items():
            if k in opts:
                opts[opts.index(k) + 1] = v
            else:
                opts.extend([k, v])

    if run_type == "votrain":
        with open(exp_config, "r") as f:
            config = yaml.load(f, yaml.loader.FullLoader)
    else:
        config = get_config(exp_config, opts)
        if seed is not None:
            config.defrost()
            config.P_SEED = seed
            config.freeze()
    execute_exp(config, run_type, wandb_run, name)


if __name__ == "__main__":
    main()
