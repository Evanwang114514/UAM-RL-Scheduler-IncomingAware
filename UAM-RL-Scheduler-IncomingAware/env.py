import csv
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class Vertiport8Env(gym.Env):
    """
    完整环境：走路 + 排队 + 飞机调度
    奖励函数：R = -ATT
    乘客数据：从 CSV 文件读取
    """

    def __init__(self, max_steps=600, mode="uniform", num_aircraft=8, spawn_interval=5,
                 passenger_file=None):
        super().__init__()
        self.max_steps = max_steps
        self.mode = mode
        self.num_aircraft = num_aircraft
        self.map_size = 40
        self.block_size = 10
        self.step_count = 0

        self.passenger_file = passenger_file
        self._all_passengers = []

        if passenger_file is not None and os.path.exists(passenger_file):
            self._load_passengers_from_file()
            self.max_passengers = len(self._all_passengers)
        else:
            self.max_passengers = 50
            if passenger_file is not None:
                print(f"乘客文件不存在: {passenger_file}，使用随机生成")

        self.vertiports = {
            0: [5, 5], 1: [35, 5], 2: [15, 15], 3: [25, 15],
            4: [15, 25], 5: [25, 25], 6: [5, 35], 7: [35, 35]
        }
        self.vertiport_list = list(self.vertiports.values())

        self.walking_remaining = []
        self.passenger_spawn_step = {}
        self.passenger_arrive_step = {}
        self.passenger_takeoff_step = {}
        self.passenger_final_step = {}

        self.passenger_origin = None
        self.passenger_dest = None
        self.current_passenger_id = -1
        self.current_spawn_step = 0
        self.current_walk_time = 0
        self.current_fly_time = 0
        self.current_final_walk = 0
        self.current_dep_vp = -1
        self.current_arr_vp = -1

        self.pending_passengers = []

        self.queues = [[] for _ in range(8)]
        self.airport_aircraft = [[] for _ in range(8)]
        self.flying_aircraft = [[] for _ in range(8)]
        self.assigned_passengers = [[] for _ in range(8)]

        self.walking_passengers = []
        self.completed_passengers = []

        self.passenger_id = 1
        self.total_spawned = 0
        self.spawn_interval = spawn_interval

        self._prev_att = 0.0
        self._prev_completed_count = 0

        self.observation_space = spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        self.action_space = spaces.Discrete(57)

    def _load_passengers_from_file(self):
        self._all_passengers = []
        if not os.path.exists(self.passenger_file):
            return
        with open(self.passenger_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._all_passengers.append({
                    'id': int(row['id']),
                    'origin': [float(row['origin_x']), float(row['origin_y'])],
                    'dest': [float(row['dest_x']), float(row['dest_y'])]
                })

    def _get_pos(self, vp):
        if isinstance(vp, list):
            return vp
        return self.vertiport_list[vp]

    def _distance(self, p1, p2):
        p1 = self._get_pos(p1)
        p2 = self._get_pos(p2)
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _generate_one_passenger(self):
        blocks = list(range(16))
        from_block = random.choice(blocks)
        row, col = from_block // 4, from_block % 4
        origin_x = col * self.block_size + random.uniform(0, self.block_size)
        origin_y = row * self.block_size + random.uniform(0, self.block_size)
        origin = [origin_x, origin_y]

        to_block = random.choice(blocks)
        while to_block == from_block:
            to_block = random.choice(blocks)
        row, col = to_block // 4, to_block % 4
        dest_x = col * self.block_size + random.uniform(0, self.block_size)
        dest_y = row * self.block_size + random.uniform(0, self.block_size)
        dest = [dest_x, dest_y]

        return origin, dest

    def _load_next_passenger(self):
        if self.total_spawned >= self.max_passengers or not self.pending_passengers:
            return False

        p = self.pending_passengers.pop(0)
        self.passenger_origin = p['origin']
        self.passenger_dest = p['dest']
        self.current_walk_time = p['walk_time_ticks']
        self.current_fly_time = p['fly_time_ticks']
        self.current_final_walk = p['final_walk_ticks']
        self.current_dep_vp = p['dep_vp']
        self.current_arr_vp = p['arr_vp']
        self.total_spawned += 1
        self.current_passenger_id = self.passenger_id
        self.current_spawn_step = self.step_count

        self.passenger_spawn_step[self.passenger_id] = self.current_spawn_step
        self.walking_remaining[self.passenger_id - 1] = self.current_walk_time
        return True

    def _start_walking(self, action_a):
        if self.passenger_origin is None:
            return

        pid = self.current_passenger_id

        if action_a == 56:
            self.passenger_final_step[pid] = self.step_count + self.current_walk_time
            self.passenger_arrive_step[pid] = self.step_count
            self.passenger_takeoff_step[pid] = self.step_count
            self.walking_remaining[pid - 1] = 0

            self.completed_passengers.append({
                'id': pid,
                'origin': self.passenger_origin,
                'dest': self.passenger_dest,
                'dep_vp': -1,
                'arr_vp': -1,
                'walk_time': self.current_walk_time,
                'wait_time': 0,
                'fly_time': 0,
                'final_walk': 0,
                'total_time': self.current_walk_time,
                'spawn_step': self.current_spawn_step,
                'arrive_airport_step': self.step_count,
                'takeoff_step': self.step_count,
                'complete_step': self.step_count + self.current_walk_time
            })

            self.passenger_id += 1
            self.passenger_origin = None
            self.passenger_dest = None
            return

        self.walking_passengers.append({
            'id': pid,
            'dep_vp': self.current_dep_vp,
            'arr_vp': self.current_arr_vp,
            'walk_remaining': self.current_walk_time,
            'walk_time': self.current_walk_time,
            'fly_time': self.current_fly_time,
            'final_walk': self.current_final_walk,
            'spawn_step': self.current_spawn_step,
            'arrive_airport_step': 0,
            'takeoff_step': 0,
            'complete_step': 0,
            'wait_time': 0,
            'total_time': 0
        })

        self.passenger_id += 1
        self.passenger_origin = None
        self.passenger_dest = None

    def _update_walking_passengers(self):
        to_remove = []
        for i, w in enumerate(self.walking_passengers):
            w['walk_remaining'] -= 1
            idx = w['id'] - 1
            if idx < len(self.walking_remaining):
                self.walking_remaining[idx] = w['walk_remaining']
            if w['walk_remaining'] <= 0:
                w['arrive_airport_step'] = self.step_count
                self.passenger_arrive_step[w['id']] = self.step_count
                self.queues[w['dep_vp']].append(w.copy())
                to_remove.append(i)

        for i in sorted(to_remove, reverse=True):
            self.walking_passengers.pop(i)

    def _pickup_passenger_from_queue(self, aircraft_id, vp_idx):
        if not self.queues[vp_idx]:
            self.airport_aircraft[vp_idx].append(aircraft_id)
            return None

        passenger_data = self.queues[vp_idx].pop(0)

        passenger_data['takeoff_step'] = self.step_count
        passenger_data['wait_time'] = self.step_count - passenger_data['arrive_airport_step']
        passenger_data['total_time'] = (
            passenger_data['walk_time'] +
            passenger_data['wait_time'] +
            passenger_data['fly_time'] +
            passenger_data['final_walk']
        )
        passenger_data['complete_step'] = self.step_count + passenger_data['fly_time'] + passenger_data['final_walk']

        self.passenger_takeoff_step[passenger_data['id']] = self.step_count
        self.completed_passengers.append(passenger_data)

        for i, w in enumerate(self.walking_passengers):
            if w['id'] == passenger_data['id']:
                self.walking_passengers.pop(i)
                break

        eta = passenger_data['fly_time']
        self.flying_aircraft[passenger_data['arr_vp']].append((aircraft_id, eta))

        return passenger_data

    def _pickup_assigned_passenger(self, aircraft_id, vp_idx):
        if not self.assigned_passengers[vp_idx]:
            self.airport_aircraft[vp_idx].append(aircraft_id)
            return None

        passenger_data = self.assigned_passengers[vp_idx].pop(0)

        passenger_data['takeoff_step'] = self.step_count
        passenger_data['wait_time'] = self.step_count - passenger_data['arrive_airport_step']
        passenger_data['total_time'] = (
            passenger_data['walk_time'] +
            passenger_data['wait_time'] +
            passenger_data['fly_time'] +
            passenger_data['final_walk']
        )
        passenger_data['complete_step'] = self.step_count + passenger_data['fly_time'] + passenger_data['final_walk']

        self.passenger_takeoff_step[passenger_data['id']] = self.step_count
        self.completed_passengers.append(passenger_data)

        for i, w in enumerate(self.walking_passengers):
            if w['id'] == passenger_data['id']:
                self.walking_passengers.pop(i)
                break

        eta = passenger_data['fly_time']
        self.flying_aircraft[passenger_data['arr_vp']].append((aircraft_id, eta))

        return passenger_data

    def _takeoff_phase(self):
        for vp in range(8):
            if not self.queues[vp] or not self.airport_aircraft[vp]:
                continue

            self.airport_aircraft[vp].sort()

            while self.queues[vp] and self.airport_aircraft[vp]:
                aircraft_id = self.airport_aircraft[vp].pop(0)
                result = self._pickup_passenger_from_queue(aircraft_id, vp)
                if result is None:
                    self.airport_aircraft[vp].append(aircraft_id)
                    break

    def _dispatch_phase(self):
        max_count = 0
        target_vp = 0
        for vp in range(8):
            count = len(self.queues[vp])
            if count > max_count:
                max_count = count
                target_vp = vp

        if max_count == 0:
            return

        all_idle = []
        for vp in range(8):
            for aircraft_id in self.airport_aircraft[vp]:
                dist = self._distance(vp, target_vp)
                all_idle.append((aircraft_id, vp, dist))

        if not all_idle:
            return

        all_idle.sort(key=lambda x: (x[2], x[0]))
        best_aircraft_id, best_vp, _ = all_idle[0]

        self.airport_aircraft[best_vp].remove(best_aircraft_id)

        if best_vp == target_vp:
            self.airport_aircraft[target_vp].append(best_aircraft_id)
        else:
            passenger_data = self.queues[target_vp].pop(0)
            passenger_data['assigned_aircraft'] = best_aircraft_id
            self.assigned_passengers[target_vp].append(passenger_data)

            eta = int(round(self._distance(best_vp, target_vp) * 3.75))
            self.flying_aircraft[target_vp].append((best_aircraft_id, eta))

    def _update_flying_aircraft(self):
        for vp in range(8):
            new_list = []
            for aircraft_id, eta in self.flying_aircraft[vp]:
                if eta <= 1:
                    if self.assigned_passengers[vp]:
                        self._pickup_assigned_passenger(aircraft_id, vp)
                    else:
                        self.airport_aircraft[vp].append(aircraft_id)
                else:
                    new_list.append((aircraft_id, eta - 1))
            self.flying_aircraft[vp] = new_list

    def _get_obs(self):
        obs = []
        max_q = max([len(q) for q in self.queues]) + 1
        obs.extend([len(q) / max_q for q in self.queues])

        if self.passenger_origin is not None:
            obs.append(self.passenger_origin[0] / self.map_size)
            obs.append(self.passenger_origin[1] / self.map_size)
            obs.append(self.passenger_dest[0] / self.map_size)
            obs.append(self.passenger_dest[1] / self.map_size)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.passenger_id = 1
        self.total_spawned = 0
        self.passenger_origin = None
        self.passenger_dest = None
        self.pending_passengers = []
        self.walking_remaining = [-1] * self.max_passengers
        self.passenger_spawn_step = {}
        self.passenger_arrive_step = {}
        self.passenger_takeoff_step = {}
        self.passenger_final_step = {}
        self.queues = [[] for _ in range(8)]
        self.airport_aircraft = [[] for _ in range(8)]
        self.flying_aircraft = [[] for _ in range(8)]
        self.assigned_passengers = [[] for _ in range(8)]
        self.walking_passengers = []
        self.completed_passengers = []

        for i in range(self.num_aircraft):
            self.airport_aircraft[i % 8].append(i)

        if self._all_passengers:
            self.max_passengers = len(self._all_passengers)
            for p in self._all_passengers:
                self.pending_passengers.append({
                    'origin': p['origin'],
                    'dest': p['dest'],
                    'walk_time_ticks': 0,
                    'fly_time_ticks': 0,
                    'final_walk_ticks': 0,
                    'dep_vp': -1,
                    'arr_vp': -1
                })
        else:
            self.max_passengers = 50
            for _ in range(self.max_passengers):
                origin, dest = self._generate_one_passenger()
                self.pending_passengers.append({
                    'origin': origin,
                    'dest': dest,
                    'walk_time_ticks': 0,
                    'fly_time_ticks': 0,
                    'final_walk_ticks': 0,
                    'dep_vp': -1,
                    'arr_vp': -1
                })

        return self._get_obs(), {}

    def step(self, action_a):
        self.step_count += 1

        self._update_walking_passengers()

        if self.passenger_origin is not None:
            self._start_walking(action_a)

        if self.step_count % self.spawn_interval == 0:
            if self.total_spawned < self.max_passengers:
                self._load_next_passenger()
            self._takeoff_phase()
            self._dispatch_phase()

        self._update_flying_aircraft()

        reward = 0.0
        total_times = []
        for cp in self.completed_passengers:
            t = cp.get('total_time', 0)
            if t > 0:
                total_times.append(t / 10.0)

        if total_times:
            att = np.mean(total_times)
            reward = -att / 10.0
            if len(self.completed_passengers) >= self.max_passengers:
                reward += 50.0

        if len(self.queues) > 0:
            queue_lengths = [len(q) for q in self.queues]
            variance = np.var(queue_lengths)
            reward -= variance * 0.001

        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}