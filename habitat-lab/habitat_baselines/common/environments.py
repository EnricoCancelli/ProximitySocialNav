#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat.utils.visualizations.maps import get_topdown_map_from_sim, colorize_draw_agent_and_fit_to_height
from habitat_sim.agent import AgentState
from habitat_baselines.common.baseline_registry import baseline_registry

import numpy as np
import matplotlib.pyplot as plt
import mathutils

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]

        if np.isnan(self._previous_measure) or self._previous_measure == float("inf"):
            print("malformed episode {} at {}".format(self._env._current_episode.scene_id, self._env._current_episode.episode_id))
            while True:
                p_pos = self._env._sim.agents[0].get_state().position
                p_rot = self._env._sim.agents[0].get_state().rotation
                nebs = self._env._sim.get_observations_at(position=p_pos + [0, 2., 0], rotation=p_rot)
                map = get_topdown_map_from_sim(self._env._sim)
                #self._env._sim.agents[0].set_state(
                 #   AgentState(
                  #  position=p_pos + [0, 10., 0],
                   # rotation=p_pos
                    #)
                #)
                map = colorize_draw_agent_and_fit_to_height({
                    "map": map,
                    "fog_of_war_mask": None,
                    "agent_map_coord": p_pos,
                    "agent_angle": p_rot
                }, 256)
                plt.imshow(map)
                plt.show()
                plt.imshow(nebs["rgb"])
                plt.show()
                plt.imsave()
            #self._env._sim.agents[0].set_state(AgentState(position=p_pos + [0, 10., 0], rotation=p_rot))

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        # debug
        if np.isnan(current_measure) or current_measure == float("inf"):
            print("previous", self._previous_measure)
            print("current", current_measure)

        if (
            observations.get("num_steps", -1.0) == -1.0
            or self._rl_config.FULL_GEODESIC_DECAY == -1.0
        ):
            reward += self._previous_measure - current_measure
        else:
            reward += (
                self._previous_measure - current_measure
            ) * max(
                0,
                (
                    1 - observations["num_steps"]
                    / self._rl_config.FULL_GEODESIC_DECAY
                )
            )

        sim = self._env._sim
        agent_pos = sim.get_agent_state().position
        if 'PROXIMITY_PENALTY' in self._rl_config:
            if sim.social_nav:
                for p in sim.people:
                    distance = np.sqrt(
                        (p.current_position[0]-agent_pos[0])**2
                        +(p.current_position[2]-agent_pos[2])**2
                    )
                    if distance < self._rl_config.get('PENALTY_RADIUS', 1.5):
                        reward -= self._rl_config.PROXIMITY_PENALTY
                        break
            elif sim.interactive_nav:
                for p in sim.object_positions:
                    distance = np.sqrt(
                        (p[0]-agent_pos[0])**2
                        +(p[2]-agent_pos[2])**2
                    )
                    if distance < self._rl_config.get('PENALTY_RADIUS', 1.5):
                        reward -= self._rl_config.PROXIMITY_PENALTY
                        break

        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        # debug
        if self._env.get_metrics().get("human_collision", False):
            reward -= self._rl_config.get('ACCIDENT_PENALTY', 0.0)

        #assert "hit_navmesh" in observations
        if observations.get("hit_navmesh", False):
            reward -= self._rl_config.COLLISION_PENALTY

        if observations.get("moving_backwards", False):
            reward -= self._rl_config.BACKWARDS_PENALTY

        if 'ang_accel' in observations:
            reward -= min(
                abs(observations['ang_accel'])
                * self._rl_config.ANG_ACCEL_PENALTY_COEFF,
                self._rl_config.MAX_ANG_ACCEL_PENALTY,
            )

        if "frontal" in observations:
            in_encounter = False
            for k, v in observations["frontal"].items():
                if k == "l_vel":
                    continue
                if v["num"].item() != 0:
                    in_encounter = True
                    distance = np.sqrt(
                        (sim.people[v["pid"].item()].current_position[0] - agent_pos[0]) ** 2
                        + (sim.people[v["pid"].item()].current_position[2] - agent_pos[2]) ** 2
                    )
                    if distance < self._rl_config.get('DANGER_THRES', 1.5):
                        reward -= self._rl_config.DANGER_PENALTY
                        break
            if in_encounter:
                reward -= self._rl_config.CAUTION_PENALTY if observations["frontal"]["l_vel"] > 0.5 else 0



        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
