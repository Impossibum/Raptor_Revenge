import math
import random

import numpy as np
from typing import Any, List, Set
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
#from bubo_misc_utils import simple_physics_object_mirror
from rlgym.utils.common_values import BOOST_LOCATIONS
from itertools import chain
from random import shuffle


class AdvancedItemizer(ObsBuilder):
    def __init__(self, expanding: bool = True):
        super().__init__()
        self.expanding = expanding
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BALL = [1, 0, 0, 0, 0, 0]
        self.PLAYER = [0, 1, 0, 0, 0, 0]
        self.TEAMMATE = [0, 0, 1, 0, 0, 0]
        self.OPPONENT = [0, 0, 0, 1, 0, 0]
        self.BOOST = [0, 0, 0, 0, 1, 0]
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.inverted_boost_locations = self.boost_locations[::-1]
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.state = None

    def reset(self, initial_state: GameState):
        self.state = None
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])

    def update_boost_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(0, self.boost_timers[i]-self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            self.BALL,
            ball.position / self.POS_STD,
            ball.linear_velocity / self.VEL_STD,
            ball.angular_velocity / self.ANG_STD,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [
                math.sqrt(sum([x * x for x in ball.linear_velocity]))/2300,
                int(ball.position[2] <= 100),
                int(abs(ball.position[0]) >= 3095 or abs(ball.position[1]) >= 5000),
                0
            ],
        ]
        # 6 + (9*3=27) +7 = 40
        #return list(chain(*p))
        return tuple(chain(*p))

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray):
        p = [
            car.position / self.POS_STD,
            car.linear_velocity / self.VEL_STD,
            car.angular_velocity / self.ANG_STD,
            (ball.position - car.position) / self.POS_STD,
            (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
            car.forward(),
            car.up(),
            [
                math.sqrt(sum([x * x for x in car.linear_velocity]))/2300,
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
            prev_act,
        ]
        #return list(chain(*p))
        return tuple(chain(*p))
        # (7*3) + 5 + 8 = 34
        #return np.array(p, dtype=np.dtype(float)).flatten()

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject, _car: PlayerData, ball: PhysicsObject, teammate: bool):
        p = [
                self.TEAMMATE if teammate else self.OPPONENT,
                car.position / self.POS_STD,
                car.linear_velocity / self.VEL_STD,
                car.angular_velocity / self.ANG_STD,
                (car.position - player_car.position) / self.POS_STD,
                (car.linear_velocity - player_car.linear_velocity) / self.VEL_STD,
                (ball.position - car.position) / self.POS_STD,
                (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
                car.forward(),
                car.up(),
                [_car.boost_amount,
                    int(_car.on_ground),
                    int(_car.has_flip),
                    int(_car.is_demoed)
                ]
            ]
        # 6 + (9*3) + 4 = 37
        #return list(chain(*p))
        return tuple(chain(*p))


    def create_boost_packet(self, player_car: PhysicsObject, boost_index: int, inverted: bool):
        b_a_l = self.inverted_boosts_availability if inverted else self.boosts_availability
        loc = self.boost_locations[boost_index] if not inverted else self.inverted_boost_locations[boost_index]
        p = [
            self.BOOST,
            loc / self.POS_STD,
            [0, 0, 0],
            player_car.angular_velocity / self.ANG_STD,
            (loc - player_car.position) / self.POS_STD,
            (-player_car.linear_velocity / self.VEL_STD),
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1.0 if loc[2] == 73.0 else 0.12,
                1.0,
                0.0,
                int(b_a_l[boost_index])
             ]
        ]
        #return list(chain(*p))
        return tuple(chain(*p))

    def add_boosts_to_obs(self, obs: List, player_car: PhysicsObject, inverted: bool):

        for i in range(self.boost_locations.shape[0]):
            obs.add(self.create_boost_packet(player_car, i, inverted))

    def add_players_to_obs(self, obs: List, state: GameState, player: PlayerData, ball: PhysicsObject,
                           prev_act: np.ndarray, inverted: bool):

        player_data = self.create_player_packet(player, player.inverted_car_data if inverted else player.car_data, ball, prev_act)
        #obs.add(player_data)

        for p in state.players:
            if p.car_id == player.car_id:
                continue
            obs.add(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                          p.inverted_car_data if inverted else p.car_data, p, ball, p.team_num == player.team_num))

        return player_data

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
        else:
            inverted = False
            ball = state.ball

        obs = set()
        obs.add(self.create_ball_packet(ball))
        player_data = self.add_players_to_obs(obs, state, player, ball, previous_action, inverted)
        self.add_boosts_to_obs(obs, player.inverted_car_data if inverted else player.car_data, inverted)
        #shuffle(obs)
        _obs = []
        _obs.extend(player_data)

        for o in obs:
            _obs.extend(o)

        print(len(_obs))

        if self.expanding:
            return np.expand_dims(_obs, 0)
        return _obs


class AdvancedBullShitter(ObsBuilder):
    def __init__(self, expanding: bool = True):
        super().__init__()
        self.expanding = expanding
        self.POS_STD = 2300
        self.VEL_STD = 2300
        self.ANG_STD = 5.5
        self.BALL = [1, 0, 0, 0, 0, 0]
        self.PLAYER = [0, 1, 0, 0, 0, 0]
        self.TEAMMATE = [0, 0, 1, 0, 0, 0]
        self.OPPONENT = [0, 0, 0, 1, 0, 0]
        self.BOOST = [0, 0, 0, 0, 1, 0]
        self.dummy_player = [0] * 38
        self.boost_locations = np.array(BOOST_LOCATIONS)
        self.inverted_boost_locations = self.boost_locations[::-1]
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.boost_objs = []
        self.inverted_boost_objs = []
        self.state = None


    def reset(self, initial_state: GameState):
        self.state = None
        self.boost_timers = np.zeros(self.boost_locations.shape[0])
        self.inverted_boost_timers = np.zeros(self.boost_locations.shape[0])
        self.boosts_availability = np.zeros(self.boost_locations.shape[0])
        self.inverted_boosts_availability = np.zeros(self.boost_locations.shape[0])

    def update_boost_timers(self, state: GameState):
        current_boosts = state.boost_pads
        boost_locs = self.boost_locations

        for i in range(len(current_boosts)):
            if current_boosts[i] == self.boosts_availability[i]:
                if self.boosts_availability[i] == 0:
                    self.boost_timers[i] = max(0, self.boost_timers[i]-self.time_interval)
            else:
                if self.boosts_availability[i] == 0:
                    self.boosts_availability[i] = 1
                    self.boost_timers[i] = 0
                else:
                    self.boosts_availability[i] = 0
                    if boost_locs[i][2] == 73:
                        self.boost_timers[i] = 10.0
                    else:
                        self.boost_timers[i] = 4.0
        self.boosts_availability = current_boosts
        self.inverted_boost_timers = self.boost_timers[::-1]
        self.inverted_boosts_availability = self.boosts_availability[::-1]

    def create_ball_packet(self, ball: PhysicsObject):
        p = [
            ball.position / self.POS_STD,
            ball.linear_velocity / self.VEL_STD,
            ball.angular_velocity / self.ANG_STD,
            [
                math.sqrt(sum([x * x for x in ball.linear_velocity]))/2300,
                int(ball.position[2] <= 100),
                int(abs(ball.position[0]) >= 3095 or abs(ball.position[1]) >= 5000),
            ],
        ]
        return list(chain(*p))

    def create_player_packet(self, player: PlayerData, car: PhysicsObject, ball: PhysicsObject, prev_act: np.ndarray):
        p = [
            car.position / self.POS_STD,
            car.linear_velocity / self.VEL_STD,
            car.angular_velocity / self.ANG_STD,
            (ball.position - car.position) / self.POS_STD,
            (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
            car.forward(),
            car.up(),
            [
                np.linalg.norm(car.linear_velocity)/2300,
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
            ],
            prev_act,
        ]
        return list(chain(*p))
        #return tuple(chain(*p))
        # (7*3) + 5 + 8 = 34
        #return np.array(p, dtype=np.dtype(float)).flatten()

    def create_car_packet(self, player_car: PhysicsObject, car: PhysicsObject, _car: PlayerData, ball: PhysicsObject, teammate: bool):
        diff = car.position - player_car.position
        magnitude = np.linalg.norm(diff)
        p = [
                self.TEAMMATE if teammate else self.OPPONENT,
                car.position / self.POS_STD,
                car.linear_velocity / self.VEL_STD,
                car.angular_velocity / self.ANG_STD,
                diff / self.POS_STD,
                (car.linear_velocity - player_car.linear_velocity) / self.VEL_STD,
                (ball.position - car.position) / self.POS_STD,
                (ball.linear_velocity - car.linear_velocity) / self.VEL_STD,
                car.forward(),
                car.up(),
                [_car.boost_amount,
                    int(_car.on_ground),
                    int(_car.has_flip),
                    int(_car.is_demoed),
                    magnitude/self.POS_STD
                ]
            ]
        # 6 + (9*3) + 5 = 38
        return list(chain(*p))
        #return tuple(chain(*p))

    def create_boost_packet(self, player_car: PhysicsObject, boost_index: int, inverted: bool):
        b_a_l = self.inverted_boosts_availability if inverted else self.boosts_availability
        loc = self.boost_locations[boost_index] if not inverted else self.inverted_boost_locations[boost_index]
        diff = loc - player_car.position
        magnitude = np.linalg.norm(diff)
        p = [
            diff / self.POS_STD,  #direction
            [0 if not bool(b_a_l[boost_index]) else (1.0 if loc[2] == 73.0 else 0.12), #current boost value
             magnitude / self.POS_STD  # current distance scaled by pos std
             ]
        ]
        return list(chain(*p))
        #return tuple(chain(*p))

    def create_boost_lists(self):
        normal = []
        inverted = []

        for i in range(len(self.boosts_availability)):
            normal.append(0 if not bool(self.boosts_availability[i]) else (1.0 if self.boost_locations[i][2] == 73.0 else 0.12))
            inverted.append(0 if not bool(self.inverted_boosts_availability[i]) else (1.0 if self.inverted_boost_locations[i][2] == 73.0 else 0.12))

        self.boost_objs = normal
        self.inverted_boost_objs = inverted

    def add_boosts_to_obs(self, obs: List, player_car: PhysicsObject, inverted: bool):

        # for i in range(self.boost_locations.shape[0]):
        #     obs.extend(self.create_boost_packet(player_car, i, inverted))
        if inverted:
            obs.extend(self.inverted_boost_objs)
        else:
            obs.extend(self.boost_objs)

    def add_players_to_obs(self, obs: Set, state: GameState, player: PlayerData, ball: PhysicsObject,
                           prev_act: np.ndarray, inverted: bool):

        player_data = self.create_player_packet(player, player.inverted_car_data if inverted else player.car_data, ball, prev_act)
        b_max = 3
        o_max = 3

        if player.team_num == 0:
            b_max = 2
        else:
            o_max = 2

        b_count = 0
        o_count = 0

        for p in state.players:
            if p.car_id == player.car_id:
                continue

            if p.team_num == 0 and b_count < b_max:
                b_count += 1
            elif p.team_num == 1 and o_count < o_max:
                o_count += 1
            else:
                continue
            obs.append(self.create_car_packet(player.inverted_car_data if inverted else player.car_data,
                          p.inverted_car_data if inverted else p.car_data, p, ball, p.team_num == player.team_num))

        for _ in range(b_max - b_count):
            obs.append(self.dummy_player)

        for _ in range(o_max - o_count):
            obs.append(self.dummy_player)

        return player_data

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if state != self.state:
            self.boosts_availability = state.boost_pads
            self.inverted_boosts_availability = state.inverted_boost_pads
            self.state = state
            self.create_boost_lists()

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
        else:
            inverted = False
            ball = state.ball

        obs = []
        players_data = []
        player_dat = self.add_players_to_obs(players_data, state, player, ball, previous_action, inverted)
        obs.extend(player_dat)
        obs.extend(self.create_ball_packet(ball))
        random.shuffle(players_data)
        for p in players_data:
            obs.extend(p)
        self.add_boosts_to_obs(obs, player.inverted_car_data if inverted else player.car_data, inverted)
        #print(len(obs))
        if self.expanding:
            return np.expand_dims(obs, 0)
        return obs


if __name__ == "__main__":
    gs = GameState()
    p = PlayerData()
    p.team_num = 0
    p.car_id = 1
    pa = PlayerData()
    pa.team_num = 0
    pa.car_id = 2
    pe = PlayerData()
    pe.team_num = 1
    pe.car_id = 3
    #print(gs.players)
    gs.players.append(p)
    gs.players.append(pa)
    #gs.players.append(pa)
    #gs.players.append(pe)
    #gs.players.append(pe)
    #gs.players.append(pe)
    #print(gs.players)
    #observer = AdvancedItemizer(tick_rate=8)
    # observer = AdvancedObsPadder(team_size=3)#AdvancedObs()
    # observer.reset(gs)
    # obs = observer.build_obs(p, gs, np.zeros(8))


    a_observer = AdvancedBullShitter()
    a_observer.reset(gs)
    a_obs = a_observer.build_obs(pe, gs, np.zeros(8))
    #print(a_obs[1][0])
    # for each in a_obs[1]:
    #     print(len(each))

    #print(obs.shape, a_obs[1].shape) #(269,) (138,) (269,) (169,)


