import os
import sys
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from custom_obs import AdvancedBullShitter
from N_Parser import NectoAction
from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rl_five_reward import RLFiveReward
from kb_setter import KB_Setter
from rewards import StarterReward
import torch
torch.set_num_threads(1)

if __name__ == "__main__":
    tick_skip = 12
    fps = 120/tick_skip
    send_state = False
    if int(sys.argv[1]) == 0:
        send_state = True
    match = Match(
        game_speed=100,
        spawn_opponents=True,
        team_size=3,
        state_setter=KB_Setter(),
        obs_builder=AdvancedBullShitter(),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 30), GoalScoredCondition()],
        reward_function=StarterReward(),
        tick_skip=tick_skip
    )

    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    r = Redis(host="127.0.0.1", password=os.environ["redis"])

    RedisRolloutWorker(r, "Impossibum", match,
                       past_version_prob=0.0,
                       evaluation_prob=0.0,
                       sigma_target=2,
                       dynamic_gm=True,
                       send_obs=True,
                       streamer_mode=False,
                       send_gamestates=send_state,
                       force_paging=True,
                       auto_minimize=True,
                       local_cache_name="raptor_model_database").run()

