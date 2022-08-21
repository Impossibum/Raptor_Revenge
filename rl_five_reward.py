from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rewards import JumpTouchReward, TouchVelChange, PositiveWrapperReward, OmniBoostDiscipline, \
    OncePerStepRewardWrapper, EventReward, KickoffReward, LavaFloorReward, VelocityBallToGoalReward, \
    VelocityPlayerToBallReward, AerialTraining


class SimplifiedBaseReward(RewardFunction):
    def __init__(self, boost_weight=1.0):
        super().__init__()
        self.goal_reward = 10.0
        self.boost_weight = boost_weight
        self.ts = 12
        self.reward = None
        self.orange_count = 0
        self.blue_count = 0
        self.boost_disc_weight = self.boost_weight * ((33.3334 / (120/self.ts)) * 0.01)

    def setup_reward(self, initial_state: GameState) -> None:

        for p in initial_state.players:
            # no access to player team makes proper assignments difficult.
            # luckily this setup will work as long as teams are even
            if p.team_num == 1:
                self.orange_count += 1
            else:
                self.blue_count += 1

        self.reward = OncePerStepRewardWrapper(CombinedReward((
            EventReward(team_goal=self.goal_reward, demo=self.goal_reward/3.0, boost_pickup=self.boost_weight),
            PositiveWrapperReward(VelocityBallToGoalReward()),
            TouchVelChange(),
            JumpTouchReward(min_height=120),
            KickoffReward(boost_punish=False),
            OmniBoostDiscipline(aerial_forgiveness=True)
        ),
            (1.0, 0.075, 1.0, 3.0, 0.1, self.boost_disc_weight)))

    def reset(self, initial_state: GameState) -> None:
        if self.reward is None:
            self.setup_reward(initial_state)
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


class PersonalRewards(RewardFunction): #reward intended soley for the individual and not to be penalized by 0 sum
    def __init__(self, boost_weight=1.0):
        super().__init__()
        self.ts = 12
        self.boost_weight = boost_weight
        self.boost_disc_weight = self.boost_weight * ((33.3334 / (120/self.ts)) * 0.01)
        self.reward = CombinedReward(
            (
                LavaFloorReward(),
                VelocityPlayerToBallReward(),
                PositiveWrapperReward(AerialTraining()),
            ),
            (0.008, 0.035, 0.1))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


class RLFiveReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.blue_rewards = dict()
        self.orange_rewards = dict()
        self.prev_action_dummy = np.zeros(8)
        self.personal_rewards = dict()
        self.boost_weight = 1.0
        self.gs = None

    def gs_handler(self, state: GameState, previous_action: np.ndarray):
        if state != self.gs:
            self.gs = state
            for p in state.players:
                my_rewards = self.orange_rewards if p.team_num == 1 else self.blue_rewards
                _ = my_rewards[p.car_id].get_reward(p, state, previous_action)

    def reset(self, initial_state: GameState) -> None:
        for p in initial_state.players:
            if p.team_num == 0:
                self.blue_rewards[p.car_id] = OncePerStepRewardWrapper(SimplifiedBaseReward(boost_weight=self.boost_weight))
                self.blue_rewards[p.car_id].reset(initial_state)
            else:
                self.orange_rewards[p.car_id] = OncePerStepRewardWrapper(SimplifiedBaseReward(boost_weight=self.boost_weight))
                self.orange_rewards[p.car_id].reset(initial_state)
            self.personal_rewards[p.car_id] = PersonalRewards(boost_weight=self.boost_weight)
            self.personal_rewards[p.car_id].reset(initial_state)

    def get_enemy_average(self, enemy_team: int, state: GameState):  # previous action won't be available information
        enemies = [x for x in state.players if x.team_num == enemy_team]
        if len(enemies) < 1:
            return 0
        enemy_rewards = self.orange_rewards if enemy_team == 1 else self.blue_rewards
        reward_total = float(0)
        for e in enemies:
            reward_total += enemy_rewards[e.car_id].get_reward(e, state, self.prev_action_dummy)
        return reward_total/len(enemies)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        #self.gs_handler(state, previous_action)
        my_rewards = self.orange_rewards if player.team_num == 1 else self.blue_rewards
        base_reward = my_rewards[player.car_id].get_reward(player, state, previous_action)
        base_reward += self.personal_rewards[player.car_id].get_reward(player, state, previous_action)
        adjusted_reward = base_reward - self.get_enemy_average(1 if player.team_num == 0 else 0, state)
        return adjusted_reward



