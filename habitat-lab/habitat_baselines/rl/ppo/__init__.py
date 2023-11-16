#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

from habitat_baselines.rl.ppo.policy import Net, PointNavBaselinePolicy, Policy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ppo.belief_ppo_trainer import BeliefPPOTrainer
from habitat_baselines.rl.ppo.belief_ppo_trainer_implicit import BeliefPPOTrainerImpl
from habitat_baselines.rl.ppo.belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy, #MidLevelPolicy,
    FixedAttentionBeliefPolicy, AverageBeliefPolicy, SoftmaxBeliefPolicy,
)

import habitat_baselines.common.aux_tasks

SINGLE_BELIEF_CLASSES: Dict[str, Policy] = {
    "BASELINE": PointNavBaselinePolicy,
    "SINGLE_BELIEF": BeliefPolicy
    #"MIDLEVEL": MidLevelPolicy,
}

MULTIPLE_BELIEF_CLASSES = {
    "ATTENTIVE_BELIEF": AttentiveBeliefPolicy,
    "FIXED_ATTENTION_BELIEF": FixedAttentionBeliefPolicy,
    "AVERAGE_BELIEF": AverageBeliefPolicy,
    "SOFTMAX_BELIEF": SoftmaxBeliefPolicy,
}

POLICY_CLASSES = dict(SINGLE_BELIEF_CLASSES, **MULTIPLE_BELIEF_CLASSES)

__all__ = ["PPO", "Policy", "Net", "PointNavBaselinePolicy", "POLICY_CLASSES", "SINGLE_BELIEF_CLASSES", "MULTIPLE_BELIEF_CLASSES"]
