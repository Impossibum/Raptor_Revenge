import os
import sys
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from custom_obs import AdvancedBullShitter
from N_Parser import NectoAction
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, TimeoutError
from redis_rollout_worker import RedisRolloutWorker
from rl_five_reward import RLFiveReward
from kb_setter import KB_Setter
from rewards import StarterReward
import torch
torch.set_num_threads(1)


if __name__ == "__main__":
    tick_skip = 12
    fps = 120/tick_skip
    send_state = False
    cache_writer = False
    # if int(sys.argv[1]) == 0:
    #     send_state = True
    #     cache_writer = True
    match = Match(
        game_speed=100,
        spawn_opponents=True,
        team_size=3,
        state_setter=KB_Setter(),
        obs_builder=AdvancedBullShitter(),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=RLFiveReward(),
        tick_skip=tick_skip
    )
    # LINK TO THE REDIS SERVER YOU SHOULD HAVE RUNNING (USE THE SAME PASSWORD YOU SET IN THE REDIS
    # CONFIG)
    redis_info = {
        "host": os.environ["REDIS_HOST"] if "REDIS_HOST" in os.environ else "localhost",
        "password": os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else None,
    }

    r = Redis(host=redis_info["host"],
              password=redis_info["password"],
              retry_on_error=[ConnectionError, TimeoutError],
              retry=Retry(ExponentialBackoff(cap=10, base=1), 25))

    contributor_name = os.environ["CONTRIBUTOR_NAME"] if "CONTRIBUTOR_NAME" in os.environ else "unknown"
    worker = RedisRolloutWorker(r, contributor_name, match,
                                past_version_prob=0.0,
                                evaluation_prob=0.0,
                                sigma_target=2,
                                dynamic_gm=True,
                                send_obs=True,
                                streamer_mode=False,
                                send_gamestates=send_state,
                                force_paging=True,
                                auto_minimize=True,
                                local_cache_name="raptor_model_database",
                                #local_cache_name=None,
                                redis_info=redis_info,
                                cache_writer=cache_writer)

    try:
        worker.run()
    except KeyboardInterrupt:
        worker.env.close()
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        worker.env.close()
        sys.exit(1)

