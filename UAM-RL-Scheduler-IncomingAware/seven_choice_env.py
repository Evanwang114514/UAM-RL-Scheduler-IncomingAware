"""
seven_choice_env.py - 七选一决策环境
动作 0-6: 选择前7个最近的起降场
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env import Vertiport8Env


class SevenChoiceEnv(gym.Env):
    """
    七选一决策环境包装器
    原始动作空间: 57 -> 压缩后: 7
    状态空间: 12维 (8队列 + 4位置)
    奖励: R = -ATT
    """

    def __init__(
            self,
            passenger_file='data/passengers_800.csv',
            max_steps=12000,
            seed=None,
            mode='train'
    ):
        super().__init__()

        self.max_steps = max_steps
        self.seed = seed
        self.mode = mode
        self.passenger_file = passenger_file

        self.env = Vertiport8Env(
            max_steps=max_steps,
            passenger_file=passenger_file
        )

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (12,), dtype=np.float32
        )

        self.dest_vp = -1
        self.sorted_candidates = []

        self.step_count = 0
        self.action_counts = [0] * 7

    def _get_dest_vertiport(self, dest):
        """获取离目的地最近的降落机场"""
        vertiports = self.env.vertiport_list
        dest_distances = []
        for vp_idx, vp_pos in enumerate(vertiports):
            d = np.linalg.norm(np.array(dest) - np.array(vp_pos))
            dest_distances.append((d, vp_idx))
        dest_distances.sort(key=lambda x: x[0])
        return dest_distances[0][1]

    def _get_sorted_airports(self, origin, dest):
        """所有起降场按总时间排序（走路+飞行）"""
        vertiports = self.env.vertiport_list
        dest_vp = self._get_dest_vertiport(dest)
        candidates = []

        for vp_idx, vp_pos in enumerate(vertiports):
            walk_dist = np.linalg.norm(np.array(origin) - np.array(vp_pos))
            walk_time = walk_dist / 4.0
            fly_dist = np.linalg.norm(np.array(vp_pos) - np.array(vertiports[dest_vp]))
            fly_time = fly_dist / 16.0
            total = (walk_time + fly_time) * 10
            candidates.append((vp_idx, total))

        candidates.sort(key=lambda x: x[1])
        return candidates

    def _map_action_to_original(self, action):
        """七选一动作映射到原始动作空间 """
        origin = self.env.passenger_origin
        dest = self.env.passenger_dest

        if origin is None or dest is None:
            return 56

        dest_vp = self._get_dest_vertiport(dest)
        self.dest_vp = dest_vp

        sorted_candidates = self._get_sorted_airports(origin, dest)
        self.sorted_candidates = sorted_candidates

        if action >= len(sorted_candidates):
            action = len(sorted_candidates) - 1

        dep_vp = sorted_candidates[action][0]

        if dep_vp == dest_vp:
            for i in range(len(sorted_candidates)):
                if sorted_candidates[i][0] != dest_vp:
                    dep_vp = sorted_candidates[i][0]
                    break

        mapping = [(d, a) for d in range(8) for a in range(8) if d != a]
        for action_a, (d, a) in enumerate(mapping):
            if d == dep_vp and a == dest_vp:
                return action_a

        return 56

    def _get_state(self):
        """获取状态 (12维)"""
        queues = [len(q) for q in self.env.queues]
        max_q = max(queues) + 1
        queue_norm = [q / max_q for q in queues]

        if self.env.passenger_origin is not None:
            pos = [
                self.env.passenger_origin[0] / 40.0,
                self.env.passenger_origin[1] / 40.0,
                self.env.passenger_dest[0] / 40.0,
                self.env.passenger_dest[1] / 40.0
            ]
        else:
            pos = [0.0, 0.0, 0.0, 0.0]

        state = np.array(queue_norm + pos, dtype=np.float32)
        return state

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.action_counts = [0] * 7
        self.sorted_candidates = []

        obs, info = self.env.reset(seed=seed, options=options)
        state = self._get_state()

        return state, info

    def step(self, action):
        self.step_count += 1

        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.size == 1 else int(action[0])
        else:
            action = int(action)

        if 0 <= action < 7:
            self.action_counts[action] += 1

        original_action = self._map_action_to_original(action)
        obs, reward, done, truncated, info = self.env.step(original_action)

        state = self._get_state()
        done = done or self.step_count >= self.max_steps

        return state, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()