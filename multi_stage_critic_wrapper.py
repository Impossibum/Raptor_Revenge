from torch import nn
import torch as th
from copy import deepcopy
import numpy as np


class MultiStageCriticWrapper(nn.Module):
    def __init__(self, net: nn):
        super().__init__()
        self.net = net
        self.shared = None

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
