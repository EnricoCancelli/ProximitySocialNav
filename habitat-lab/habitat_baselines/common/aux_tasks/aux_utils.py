import abc
from typing import Dict, List, Optional

import torch
import torch.nn as nn

ACTION_EMBEDDING_DIM = 4

def subsampled_mean(x, p: float=0.1):
    return torch.masked_select(x, torch.rand_like(x) < p).mean()


class RolloutAuxTask(nn.Module):
    r""" Rollout-based self-supervised auxiliary task base class.
    """

    def __init__(self, cfg, aux_cfg, task_cfg, device, hidden_size=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.aux_cfg = aux_cfg
        self.task_cfg = task_cfg  # Mainly tracked for actions
        self.device = device
        #self.aux_hidden_size = hidden_size if hidden_size is not None else cfg.hidden_size
        #self.loss_factor = getattr(aux_cfg, "loss_factor", 0.1) # absent for dummy
        #self.subsample_rate = getattr(aux_cfg, "subsample_rate", 0.1)
        #self.strategy = getattr(aux_cfg, "sample", "random")

    def forward(self):
        raise NotImplementedError

    @torch.jit.export
    @abc.abstractmethod
    def get_loss(self,
        observations: Dict[str, torch.Tensor],
        actions,
        sensor_embeddings: Dict[str, torch.Tensor],
        final_belief_state,
        belief_features,
        metrics: Dict[str, torch.Tensor],
        n: int,
        t: int,
        env_zeros: List[List[int]]
    ):
        pass

    @staticmethod
    def get_required_sensors(*args):
        # Encoders required to function (not habitat sensors, those are specified elsewhere)
        return []

    def masked_sample_and_scale(self, x, mask: Optional[torch.Tensor]=None):
        if mask is not None:
            x = torch.masked_select(x, mask)
        # strat_name = getattr(cfg, "sample", "random")
        # sampler = SUBSAMPLER.get(self.strategy, "random")
        # Drop support for hard mining
        return subsampled_mean(x, p=self.subsample_rate) * self.loss_factor
