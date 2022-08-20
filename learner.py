import os
import wandb
import torch.jit
import numpy as np
from torch.nn import Linear, Sequential, Mish
from redis import Redis
from custom_obs import AdvancedBullShitter
from N_Parser import NectoAction
from multi_stage_agent import MultiStageAgent
from multi_stage_discrete_policy import MultiStageDiscretePolicy
from multi_stage_critic_wrapper import MultiStageCriticWrapper
from shared_ppo import SharedPPO
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall
from torch.nn.init import xavier_uniform_
from rl_five_reward import RLFiveReward
from pathlib import Path
from rewards import StarterReward

def init_parameters(model):
    r"""Initiate parameters in the transformer model. Taken from PyTorch Transformer impl"""
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)

def get_latest_model_path():
    run_path = "raptor_repo/"
    paths = sorted(Path(run_path).iterdir(), key=os.path.getmtime, reverse=True)
    latest_run_path = str(paths[0])
    paths = sorted(Path(latest_run_path).iterdir(), key=os.path.getmtime, reverse=True)
    return str(paths[0])+"\\checkpoint.pt"


if __name__ == "__main__":
    run_id = "8csek9cp"
    # run_id = None
    max_obs_size = 240
    wandb.login(key=os.environ["WANDB_KEY"])

    def obs():
        return AdvancedBullShitter()

    def rew():
        return RLFiveReward()

    def act():
        return NectoAction()


    stat_trackers = [
        Speed(), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(), DistToBall()
    ]

    frame_skip = 12
    half_life_seconds = 8
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    print(f"_gamma is: {gamma}")
    # weights frozen @ 3600 * 100k steps ending 3760
    # big epochs low lr starting 3760
    config = dict(
        actor_lr=5e-5,
        #actor_lr=0.0,
        critic_lr=2e-4,
        shared_lr=5e-5,
        #shared_lr=0.0,
        n_steps=1_000_000,
        batch_size=100_000,
        minibatch_size=50_000,
        epochs=30,
        gamma=gamma,
        save_every=20, #20
        model_every=60, #60
        ent_coef=0.01,
    )

    critic = Sequential(
        Linear(512, 512),
        Mish(),
        Linear(512, 512),
        Mish(),
        Linear(512, 512),
        Mish(),
        Linear(512, 512),
        Mish(),
        Linear(512, 1)
    )
    init_parameters(critic)
    critic = MultiStageCriticWrapper(critic)

    shared = Sequential(
        Linear(max_obs_size, 512),
        Mish(),
        Linear(512, 512),
        Mish(),
        Linear(512, 512),
        Mish()
    )
    init_parameters(shared)

    actor = MultiStageDiscretePolicy(Sequential(
        Linear(512, 512),
        Mish(),
        Linear(512, 512),
        Mish(),
        Linear(512, 90)),
        (90,))
    init_parameters(actor)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": config["actor_lr"]},
        {"params": critic.parameters(), "lr": config["critic_lr"]},
        {"params": shared.parameters(), "lr": config["shared_lr"]},
    ])

    # PPO REQUIRES AN ACTOR/CRITIC AGENT
    agent = MultiStageAgent(actor=actor, critic=critic, shared=shared, optimizer=optim)
    model_parameters = filter(lambda p: p.requires_grad, agent.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"There are {params} trainable parameters")

    count = 0
    while count < 5:
        logger = wandb.init(project="Raptor", entity="impossibum", id=run_id)
        redis = Redis(password=os.environ["redis"])
        rollout_gen = RedisRolloutGenerator("Raptor", redis, obs, rew, act,
                                            logger=logger,
                                            save_every=config["save_every"],
                                            model_every=config["model_every"],
                                            clear=run_id is None,
                                            max_age=1,
                                            stat_trackers=stat_trackers)

        alg = SharedPPO(
            rollout_gen,
            agent,
            ent_coef=config["ent_coef"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            minibatch_size=config["minibatch_size"],
            epochs=config["epochs"],
            gamma=config["gamma"],
            logger=logger,
            device="cuda",
        )
        try:
            if run_id is not None:
                alg.load(get_latest_model_path())

            alg.run(iterations_per_save=config["save_every"], save_dir="raptor_repo")
        except Exception as e:
            count += 1
            print(f"ERROR! : {str(e)}")
            print(f"error count: {count}")


input("Press Enter")
