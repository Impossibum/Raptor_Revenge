import torch as th
from torch import nn
from rocket_learn.agent.policy import Policy
import numpy as np


class MultiStageAgent(nn.Module):
    def __init__(self, actor: Policy, critic: nn, shared: nn.Module, optimizer: th.optim.Optimizer):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.shared = shared
        self.optimizer = optimizer
        self.update_modules()

    def update_modules(self):
        self.actor.set_modules(self.shared)
        self.critic.set_modules(self.shared)

    def pre_forward_processing(self, obs):
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        elif isinstance(obs, tuple):
            obs = tuple(o if isinstance(o, th.Tensor) else th.from_numpy(o).float() for o in obs)
        return self.shared(obs)

    def forward(self, *args, **kwargs):
        x = self.pre_forward_processing(args)
        return self.actor(x, pre_processed=True), self.critic(x, pre_processed=True)
