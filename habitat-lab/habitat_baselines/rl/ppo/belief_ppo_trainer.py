#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import glob
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import json
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, VectorEnv, logger
from habitat.core.spaces import ActionSpace, EmptySpace
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.utils.common import is_fp16_supported, \
    is_fp16_autocast_supported
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)

from habitat_baselines.rl.ppo.policy import Net, PointNavBaselinePolicy, Policy
from habitat_baselines.rl.ppo.belief_policy import (
    BeliefPolicy, AttentiveBeliefPolicy,  # MidLevelPolicy,
    FixedAttentionBeliefPolicy, AverageBeliefPolicy, SoftmaxBeliefPolicy,
)

SINGLE_BELIEF_CLASSES: Dict[str, Policy] = {
    "BASELINE": PointNavBaselinePolicy,
    "SINGLE_BELIEF": BeliefPolicy
    # "MIDLEVEL": MidLevelPolicy,
}

MULTIPLE_BELIEF_CLASSES = {
    "ATTENTIVE_BELIEF": AttentiveBeliefPolicy,
    "FIXED_ATTENTION_BELIEF": FixedAttentionBeliefPolicy,
    "AVERAGE_BELIEF": AverageBeliefPolicy,
    "SOFTMAX_BELIEF": SoftmaxBeliefPolicy,
}

POLICY_CLASSES = dict(SINGLE_BELIEF_CLASSES, **MULTIPLE_BELIEF_CLASSES)

import yaml

from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_to_velocity_control,
    batch_obs,
    generate_video,
)

from habitat_baselines.utils.env_utils import construct_envs


@baseline_registry.register_trainer(name="belief-ddppo")
class BeliefPPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm with aux losses
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO  # TODO: maybe modify PPO
    actor_critic: Policy  # TODO: maybe change policy

    def __init__(self, config=None, runtype='train', wandb_ctx=None):
        if runtype == 'train':
            resume_state = load_resume_state(config)
            self.OVERRIDE_TOTAL_NUM_STEPS = None
            if resume_state is not None:
                if 'OVERRIDE' in config:
                    self.OVERRIDE_NUM_CHECKPOINTS = config.OVERRIDE.NUM_CHECKPOINTS
                    self.OVERRIDE_TOTAL_NUM_STEPS = config.OVERRIDE.TOTAL_NUM_STEPS
                    config = resume_state["config"]
                    config.defrost()
                    config.NUM_CHECKPOINTS = self.OVERRIDE_NUM_CHECKPOINTS
                    config.TOTAL_NUM_STEPS = self.OVERRIDE_TOTAL_NUM_STEPS
                    config.freeze()
                else:
                    config = resume_state["config"]

        super().__init__(config)
        print("init_num_up:", self.num_updates_done)
        self.wandb_ctx = wandb_ctx
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distirbuted if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

        self.discrete_actions = (
            self.config.TASK_CONFIG.TASK.ACTIONS.VELOCITY_CONTROL.DISCRETE_ACTIONS
        )
        if (
            config.RL.POLICY.action_distribution_type == "gaussian"
            or len(self.discrete_actions) > 0
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["VELOCITY_CONTROL"]
            config.freeze()

        if self.config.RL.fp16_mode not in ("off", "autocast", "mixed"):
            raise RuntimeError(
                f"Unknown fp16 mode '{self.config.RL.fp16_mode}'"
            )

        if self.config.RL.fp16_mode != "off" and not torch.cuda.is_available():
            logger.warn(
                "FP16 requires CUDA but CUDA is not available, setting to off"
            )

        self._fp16_mixed = self.config.RL.fp16_mode == "mixed"
        self._fp16_autocast = self.config.RL.fp16_mode == "autocast"

        if self._fp16_mixed and not is_fp16_supported():
            raise RuntimeError(
                "FP16 requires PyTorch >= 1.6.0, please update your PyTorch"
            )

        if self._fp16_autocast and not is_fp16_autocast_supported():
            raise RuntimeError(
                "FP16 autocast requires PyTorch >= 1.7.1, please update your PyTorch"
            )

    def _setup_aux_tasks(self):
        raise NotImplementedError()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config, aux_tasks) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        assert self.config.RL.POLICY.name in POLICY_CLASSES
        policy = POLICY_CLASSES[self.config.RL.POLICY.name]

        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)

        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        from gym.spaces import Dict, Box
        if self.config.TASK_CONFIG.SIMULATOR.get('PEOPLE_MASK', False):
            d_space = {
                'depth': Box(
                    low=0., high=1., shape=(
                        self.envs.observation_spaces[0].spaces['depth'].shape[
                            0],
                        self.envs.observation_spaces[0].spaces['depth'].shape[
                            1],
                        2,
                    )
                ),
                'pointgoal_with_gps_compass':
                    self.envs.observation_spaces[0].spaces[
                        'pointgoal_with_gps_compass']
            }
            for k in self.envs.observation_spaces[0].spaces.keys():
                if k != depth:
                    d_space[k] = self.envs.observation_spaces[0].spaces[k]
            observation_space = Dict(d_space)
        else:
            # observation_space = Dict(
            # {
            # 'rgb': self.envs.observation_spaces[0].spaces['rgb'],
            # 'depth': self.envs.observation_spaces[0].spaces['depth'], #.spaces.keys(),
            # 'pointgoal_with_gps_compass': self.envs.observation_spaces[0].spaces['pointgoal_with_gps_compass']
            # }
            # )

            #pass
            #debug
            observation_space = Dict({
                k: v for k, v in observation_space.spaces.items() if k != "frontal" #and k != "rgb"
            })
            #print(observation_space)

        # debug
        #assert 'rgb' not in self.envs.observation_spaces[0].spaces
        self.actor_critic = policy(
            observation_space=observation_space,
            action_space=self.policy_action_space,
            hidden_size=ppo_cfg.hidden_size,
            aux_tasks=aux_tasks,
            # TODO: maybe not necessary
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            num_tasks=len(self.config.RL.AUX_TASKS.tasks),
            # TODO unecessary
            additional_sensors=[],
            embed_goal=False,
            device=self.device,
            config=self.config.RL.POLICY
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        # TODO: revise if you want to use it
        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            # TODO: remember to change this if modified PPO
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            aux_loss_coef=ppo_cfg.aux_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            # TODO: add parameters for aux tasks
            aux_tasks=aux_tasks,
            aux_cfg=self.config.RL.AUX_TASKS
        )

    def _init_auxiliary_tasks(self, is_eval=False, observation_space=None):
        aux_cfg = self.config.RL.AUX_TASKS
        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG.TASK
        aux_task_strings = [task.lower() for task in aux_cfg.tasks]

        aux_counts = {}
        for i, x in enumerate(aux_task_strings):
            if x in aux_counts:
                aux_task_strings[i] = f"{aux_task_strings[i]}_{aux_counts[x]}"
                aux_counts[x] += 1
            else:
                aux_counts[x] = 1

        logger.info("Auxiliary tasks: {}".format(aux_task_strings))

        num_recurrent_memories = 1
        if self.config.RL.POLICY.name in MULTIPLE_BELIEF_CLASSES:
            num_recurrent_memories = len(aux_cfg.tasks)

        # TODO: check that it's executed only on train
        init_aux_tasks = []
        if not is_eval:
            task_classes = baseline_registry.get_aux_tasks(aux_cfg)
            for i, task in enumerate(aux_cfg.tasks):
                t_class = task_classes[i]
                task_module = t_class(
                    ppo_cfg, aux_cfg[task], task_cfg, self.device,
                    observation_space=observation_space
                ).to(self.device)
                init_aux_tasks.append(task_module)

        self.aux_names = aux_task_strings
        return init_aux_tasks, num_recurrent_memories, aux_task_strings

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.config.RL.POLICY.action_distribution_type == "gaussian":
            self.policy_action_space = ActionSpace(
                {
                    "linear_velocity": EmptySpace(),
                    "angular_velocity": EmptySpace(),
                }
            )
            action_shape = 2
            discrete_actions = False
        else:
            if len(self.discrete_actions) > 0:
                self.policy_action_space = ActionSpace(
                    {
                        str(i): EmptySpace()
                        for i in range(len(self.discrete_actions))
                    }
                )
            else:
                self.policy_action_space = self.envs.action_spaces[0]
            action_shape = 1
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        # initing auxiliary tasks
        aux_cfg = self.config.RL.AUX_TASKS
        # TODO: check
        init_aux_tasks, num_recurrent_memories, aux_task_strings = \
            self._init_auxiliary_tasks(observation_space=self.obs_space)
        # TODO: check

        self._setup_actor_critic_agent(ppo_cfg, init_aux_tasks)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
            # multiplememories
            num_recurrent_memories=num_recurrent_memories
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        #TODO: debug
        #print(observations[0])
        #print({k: v.shape for k, v in observations[0].items()})
        #raise Exception("")
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """

        def _cast(t: torch.Tensor):
            if t.dtype == torch.float16:
                return t.to(dtype=torch.float32)
            else:
                return t

        checkpoint = {
            "state_dict": {
                k: _cast(v) for k, v in self.agent.state_dict().items()
            },
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                        v
                    ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if self.config.RL.POLICY.action_distribution_type == "gaussian":
                step_action = action_to_velocity_control(
                    act, num_steps=self.num_steps_done
                )
            elif len(self.discrete_actions) > 0:
                act2 = torch.tensor(
                    self.discrete_actions[act.item()],
                    device='cpu',
                )
                step_action = action_to_velocity_control(
                    act2, num_steps=self.num_steps_done
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][
            env_slice] += current_ep_reward.where(done_masks,
                                                  current_ep_reward.new_zeros(
                                                      ()))  # type: ignore
        self.running_episode_stats["count"][
            env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks,
                                                                v.new_zeros(
                                                                    ()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        #print("update: {}".format(self.running_episode_stats["count"]))
        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        #print("updating?")
        value_loss, action_loss, dist_entropy, aux_loss, aux_entropy, aux_weights, total_loss = self.agent.update(
            self.rollouts
        )
        #print("yes")

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
            aux_loss,
            aux_entropy,
            aux_weights,
            total_loss
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        self.count_steps_delta = count_steps_delta
        #print("update: {}".format(self.window_episode_stats["count"]))
        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)
        #print(self.window_episode_stats["human_collision"])
        #print(deltas["human_collision"])
        mean_ep = (self.count_steps_delta*len(self.window_episode_stats["count"]))/deltas["count"]
        #raise Exception("")
        #  assert False

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        metrics["mean_ep_length"] = mean_ep

        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        metrics["reward"]=deltas["reward"] / deltas["count"]
        total_metrics = dict(metrics, **losses)
        if self.wandb_ctx is not None:
            self.wandb_ctx.log(total_metrics)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
                   rollout_step
                   >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
               ) and int(self.num_rollouts_done_store.get("num_done")) >= (
                   self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
               )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            if self.OVERRIDE_TOTAL_NUM_STEPS is None:
                self._last_checkpoint_percent = requeue_stats[
                    "_last_checkpoint_percent"
                ]
            else:
                self._last_checkpoint_percent = self.percent_done()
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]
            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                # print("collecting")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # print("updating")
                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                    aux_task_losses,
                    aux_dist_entropy,
                    aux_weights,
                    total_loss
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                print("num_update:", self.num_updates_done)

                self.num_updates_done += 1
                aux_losses_dict_flattened = {
                    self.aux_names[i]: loss for i, loss in enumerate(aux_task_losses)
                }
                aux_losses_dict_flattened["aux_total"] = sum(aux_task_losses)

                print("initiating coalescence")
                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss, dist_entropy=dist_entropy, total_loss=total_loss,
                         **aux_losses_dict_flattened),
                    count_steps_delta,
                )
                print("done")

                #print("training logged?")
                self._training_log(writer, losses, prev_time)
                #print("yes")
                # checkpoint model
                # print("saving")
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        #print
        print(checkpoint_path)

        assert os.path.isfile("ckp_cache.yml")
        with open("ckp_cache.yml", "w") as f:
            yaml.dump({"ckp_num": int(checkpoint_path.split(".")[-2])}, f)

        #global CKP_NUM
        #CKP_NUM = checkpoint_path.split(".")[-2]

        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Skip this checkpoint if a json result file for it already exists
        if self.config.JSON_DIR != '':
            json_path = os.path.join(
                self.config.JSON_DIR,
                f"{os.path.basename(checkpoint_path[:-4]).replace('.', '_')}.json",
            )
            if os.path.isfile(json_path):
                return

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            if config.TASK_CONFIG.TASK.TYPE == 'InteractiveNav-v0':
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            else:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append(
                    "SOCIAL_TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if self.config.RL.POLICY.action_distribution_type == "gaussian":
            self.policy_action_space = ActionSpace(
                {
                    "linear_velocity": EmptySpace(),
                    "angular_velocity": EmptySpace(),
                }
            )
            action_shape = 2
            action_type = torch.float
        else:
            if len(self.discrete_actions) > 0:
                self.policy_action_space = ActionSpace(
                    {
                        str(i): EmptySpace()
                        for i in range(len(self.discrete_actions))
                    }
                )
            else:
                self.policy_action_space = self.envs.action_spaces[0]
            action_shape = 1
            action_type = torch.long

        #loading auxes
        init_aux_tasks, num_recurrent_memories, aux_task_strings = \
            self._init_auxiliary_tasks(observation_space=self.obs_space)

        self._setup_actor_critic_agent(ppo_cfg, init_aux_tasks)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        #debug
        #torch.manual_seed(config.P_SEED) #todo42 for aux

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        if num_recurrent_memories == 1:
            rec_dim = (
                self.config.NUM_ENVIRONMENTS,
                self.actor_critic.net.num_recurrent_layers,
                ppo_cfg.hidden_size,
            )

        else:
            rec_dim = (
                self.config.NUM_ENVIRONMENTS,
                self.actor_critic.net.num_recurrent_layers,
                num_recurrent_memories,
                ppo_cfg.hidden_size,
            )

        test_recurrent_hidden_states = torch.zeros(*rec_dim, device=self.device)
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        all_episode_stats = {}
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    #deterministic=False,
                    deterministic=True,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if self.config.RL.POLICY.action_distribution_type == "gaussian":
                step_data = [
                    action_to_velocity_control(a)
                    for a in actions.to(device="cpu")
                ]
            elif len(self.discrete_actions) > 0:
                step_data = [
                    action_to_velocity_control(
                        torch.tensor(
                            self.discrete_actions[a.item()],
                            device='cpu',
                        )
                    )
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    episode_stats['num_steps'] = len(rgb_frames[i])

                    all_episode_stats[
                        current_episodes[i].episode_id
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                            # fps=30,  #maybe works
                            wandb_ctx=self.wandb_ctx
                        )

                    rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)
                else:
                    rgb_frames[i].append(None)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        # with open(checkpoint_path+'.txt', 'w') as f:
        #     f.write(f"{step_id}\n")
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")
            # f.write(f"Average episode {k}: {v:.4f}\n")

        # Save JSON file
        all_episode_stats['agg_stats'] = aggregated_stats
        json_dir = self.config.JSON_DIR
        if json_dir != '':
            os.makedirs(json_dir, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(all_episode_stats, f)

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        tot_met = {k: v for k, v in aggregated_stats.items()}
        tot_met["step"] = step_id
        print(tot_met)
        self.wandb_ctx.log(tot_met)

        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

        # End the script when all checkpoints have been evaluated
        if len(
            glob.glob(os.path.join(json_dir, '*.json'))
        ) == self.config.NUM_CHECKPOINTS:
            exit()
