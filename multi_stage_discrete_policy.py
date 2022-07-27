from typing import Optional, List, Tuple
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from rocket_learn.agent.discrete_policy import DiscretePolicy
from copy import deepcopy


class MultiStageDiscretePolicy(DiscretePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = None
        self.shared = None
        self.tgt = th.tensor((37, 512))

    def forward(self, obs, pre_processed=False):
        if not pre_processed:
            # if isinstance(obs, np.ndarray):
            #     obs = th.from_numpy(obs).float()
            # elif isinstance(obs, tuple):
            #     obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)
            obs = self.shared(obs)
        return self.net(obs)

    def set_modules(self, shared_layers):
        self.shared = deepcopy(shared_layers)

    def get_action_distribution(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)

        obs = self.shared(obs)
        logits = self(obs, pre_processed=True)
        if isinstance(logits, th.Tensor):
            logits = (logits,)

        max_shape = max(self.shape)
        logits = th.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in logits
            ],
            dim=1
        )
        return Categorical(logits=logits)
