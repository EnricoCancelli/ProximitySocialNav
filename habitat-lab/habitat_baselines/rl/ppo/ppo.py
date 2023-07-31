#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: Policy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        aux_loss_coef: float = 0.0,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
        aux_tasks=[],
        aux_cfg=None,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        params = list(actor_critic.parameters())
        self.aux_tasks=[]
        if aux_cfg:
            self.aux_cfg = aux_cfg
            self.aux_tasks = aux_tasks
        for aux_t in aux_tasks:
            params += list(aux_t.parameters())

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, params)),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1]
            - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Tuple[
        float, float, float, List[float], float, List[float]]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        total_loss_epoch = 0.0
        aux_losses_epoch = [0] * len(self.aux_tasks)
        aux_entropy_epoch = 0
        aux_weights_epoch = [0] * len(self.aux_tasks)
        aux_dist_entropy = None
        aux_weights = None

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                if len(self.aux_tasks)>0:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        final_rnn_state,
                        rnn_features,
                        individual_rnn_features,
                        aux_dist_entropy,
                        aux_weights,
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                    )
                else:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _
                    ) = self._evaluate_actions(
                        batch["observations"],
                        batch["recurrent_hidden_states"],
                        batch["prev_actions"],
                        batch["masks"],
                        batch["actions"],
                    )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                total_aux_loss = 0
                aux_losses = []
                if len(self.aux_tasks) > 0:
                    aux_raw_losses = self.actor_critic.evaluate_aux_losses(batch, final_rnn_state, rnn_features, individual_rnn_features)
                    aux_losses = torch.stack(aux_raw_losses)
                    total_aux_loss = torch.sum(aux_losses, dim=0)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    + total_aux_loss * self.aux_loss_coef
                    - dist_entropy * self.entropy_coef
                )

                if aux_dist_entropy is not None:
                    # TODO: maybe take the mean of the entropy, since also dist_entropy is averaged on line 150
                    total_loss -= aux_dist_entropy * self.aux_cfg.entropy_coef

                #debug
                if np.isnan(total_loss.item()):
                    print("total_loss is nan")

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                total_loss_epoch += total_loss.item()

                if aux_dist_entropy is not None:
                    aux_entropy_epoch += aux_dist_entropy.item()
                for i, aux_loss in enumerate(aux_losses):
                    aux_losses_epoch[i] += aux_loss.item()
                if aux_weights is not None:
                    for i, aux_weight in enumerate(aux_weights):
                        aux_weights_epoch[i] += aux_weight.item()

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        total_loss_epoch /= num_updates
        for i, aux_loss in enumerate(aux_losses):
            aux_losses_epoch[i] /= num_updates
        if aux_weights is not None:
            for i, aux_weight in enumerate(aux_weights):
                aux_weights_epoch[i] /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, aux_losses_epoch, aux_entropy_epoch, aux_weights_epoch, total_loss_epoch

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass
