#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Categorical
from typing import Type

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.aux_tasks.aux_utils import subsampled_mean, ACTION_EMBEDDING_DIM, RolloutAuxTask


@baseline_registry.register_aux_task(name="Risk")
class RiskEstimation(RolloutAuxTask):

    def __init__(self, cfg, aux_cfg, task_cfg, device, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        assert "RISK_SENSOR" in task_cfg.SENSORS, \
            "RiskEstimation requires RiskSensor, include it in SENSOR variable"
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size, 1), nn.Sigmoid()
        )
        self.loss = nn.MSELoss(reduction="none")

        if len(task_cfg.POSSIBLE_ACTIONS) == 1 and task_cfg.POSSIBLE_ACTIONS[0] == "VELOCITY_CONTROL":
            self.action_embedder = nn.Linear(2, ACTION_EMBEDDING_DIM)
            self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)
        else:
            self.action_embedder = nn.Embedding(num_actions + 1, ACTION_EMBEDDING_DIM)
            self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = self.aux_cfg.num_steps # up to t
        assert 0 < k <= t, "CPC requires prediction range to be in (0, t]"
        t_hat = t-k+1

        belief_features = belief_features[:t_hat].view(t_hat*n, -1).unsqueeze(0)

        risk_values = observations["risk"].view(t, n, 1)

        # TODO: squeeze here
        actions = torch.squeeze(actions, dim=-1)
        action_embedding = self.action_embedder(actions) # t n -1
        action_seq = action_embedding.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2).view(k, t_hat*n, ACTION_EMBEDDING_DIM)

        risk_seq = risk_values.unfold(dimension=0, size=k, step=1).permute(3, 0, 1, 2).view(k, t_hat*n, 1)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        # (k, t_hat*n, hidden)

        # Targets: predict k steps for each starting timestep
        pred_risk = self.classifier(out_all.view(k*t_hat*n, -1))\
            .view(k, t_hat*n, -1)
        losses = self.loss(pred_risk, risk_seq).permute(1, 0, 2).view(t_hat, n, k, -1)

        valid_modeling_queries = torch.ones(
            t_hat, n, 1, 1, device=self.device, dtype=torch.bool
        )
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                valid_modeling_queries[z - k: z, env, :] = False

        losses = torch.masked_select(losses, valid_modeling_queries)

        subsampled_loss = subsampled_mean(losses, p=self.aux_cfg.subsample_rate)

        return subsampled_loss * self.aux_cfg.loss_factor


@baseline_registry.register_aux_task
class SocialCompass(RolloutAuxTask):

    def __init__(self, cfg, aux_cfg, task_cfg, device, observation_space, **kwargs):
        super().__init__(cfg, aux_cfg, task_cfg, device, observation_space, **kwargs)
        num_actions = len(task_cfg.POSSIBLE_ACTIONS)
        assert "SOCIAL_COMPASS" in task_cfg.SENSORS, \
            "SocialCompass requires SocialCompassSensor, include it in SENSOR variable"
        self.num_bins = observation_space.spaces["social_compass"].shape[0]
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size, self.num_bins), nn.Sigmoid()
        )
        self.loss = nn.MSELoss(reduction="none")

        if len(task_cfg.POSSIBLE_ACTIONS) == 1 and task_cfg.POSSIBLE_ACTIONS[
            0] == "VELOCITY_CONTROL":
            self.action_embedder = nn.Linear(2, ACTION_EMBEDDING_DIM)
            self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)
        else:
            self.action_embedder = nn.Embedding(num_actions + 1,
                                                ACTION_EMBEDDING_DIM)
            self.query_gru = nn.GRU(ACTION_EMBEDDING_DIM, cfg.hidden_size)

    def get_loss(self, observations, actions, vision, final_belief_state, belief_features, n, t, env_zeros):
        k = self.aux_cfg.num_steps  # up to t
        assert 0 < k <= t, "CPC requires prediction range to be in (0, t]"
        t_hat = t - k + 1

        belief_features = belief_features[:t_hat].view(t_hat * n,
                                                       -1).unsqueeze(0)

        compass_values = observations["social_compass"].view(t, n, -1)

        action_embedding = self.action_embedder(actions)  # t n -1
        action_seq = action_embedding.unfold(dimension=0, size=k, step=1).\
            permute(3, 0, 1, 2).view(k, t_hat * n, ACTION_EMBEDDING_DIM)

        compass_seq = compass_values.unfold(dimension=0, size=k, step=1).\
            permute(3, 0, 1, 2).view(k, t_hat * n, -1)

        # for each timestep, predict up to k ahead -> (t,k) is query for t + k (+ 1) (since k is 0-indexed) i.e. (0, 0) -> query for 1
        out_all, _ = self.query_gru(action_seq, belief_features)
        # (k, t_hat*n, hidden)

        # Targets: predict k steps for each starting timestep
        pred_compass = self.classifier(out_all.view(k * t_hat * n, -1)) \
            .view(k, t_hat * n, -1)
        losses = self.loss(pred_compass, compass_seq).permute(1, 0, 2).view(t_hat, n,
                                                                      k, -1)

        valid_modeling_queries = torch.ones(
            t_hat, n, 1, 1, device=self.device, dtype=torch.bool
        )
        for env in range(n):
            has_zeros_batch = env_zeros[env]
            for z in has_zeros_batch:
                valid_modeling_queries[z - k: z, env, :] = False

        losses = torch.masked_select(losses, valid_modeling_queries)

        subsampled_loss = subsampled_mean(losses,
                                          p=self.aux_cfg.subsample_rate)

        return subsampled_loss * self.aux_cfg.loss_factor
