"""
test.py - 测试老PPO（12维）
"""

import numpy as np
import csv
import os
import argparse
import torch
from env import Vertiport8Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

VERTIPORTS = [[5, 5], [35, 5], [15, 15], [25, 15], [15, 25], [25, 25], [5, 35], [35, 35]]

CAV_SPEED_KPH = 20.0
EVTOL_SPEED_KPH = 120.0

WALK_SPEED = CAV_SPEED_KPH / 60.0
FLIGHT_SPEED = EVTOL_SPEED_KPH / 60.0


def distance_km(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def get_action(dep_vp, arr_vp):
    mapping = [(d, a) for d in range(8) for a in range(8) if d != a]
    for action_a, (d, a) in enumerate(mapping):
        if d == dep_vp and a == arr_vp:
            return action_a
    return 56


def get_shortest_action(origin, dest):
    ground_dist = distance_km(origin, dest)
    ground_time = ground_dist / WALK_SPEED

    mapping = [(d, a) for d in range(8) for a in range(8) if d != a]
    best_action = 56
    best_total = ground_time
    best_type = '地面'
    best_dep = -1
    best_arr = -1
    best_g1 = best_fly = best_g2 = 0

    for action_a, (dep, arr) in enumerate(mapping):
        dep_pos = np.array(VERTIPORTS[dep])
        arr_pos = np.array(VERTIPORTS[arr])

        g1 = distance_km(origin, dep_pos) / WALK_SPEED
        fly = distance_km(dep_pos, arr_pos) / FLIGHT_SPEED
        g2 = distance_km(arr_pos, dest) / WALK_SPEED
        total = g1 + fly + g2

        if total < best_total:
            best_total = total
            best_action = action_a
            best_type = f'飞机({dep}->{arr})'
            best_dep, best_arr = dep, arr
            best_g1, best_fly, best_g2 = g1, fly, g2

    info = {
        'type': best_type,
        'total': best_total,
        'dep': best_dep,
        'arr': best_arr,
        'action': best_action,
        'is_ground': best_action == 56,
        'g1': best_g1,
        'fly': best_fly,
        'g2': best_g2
    }
    return best_action, info


def load_passengers_from_csv(filename):
    passengers = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            passengers.append({
                'id': int(row['id']),
                'origin': [float(row['origin_x']), float(row['origin_y'])],
                'dest': [float(row['dest_x']), float(row['dest_y'])]
            })
    return passengers


def test_rl_model(model_path, passenger_file='data/passengers_800_test.csv', n_passengers=800, max_ticks=12000):
    print("=" * 120)
    print(f"PPO-12测试 - {n_passengers}人 (测试集)")
    print("=" * 120)
    print(f"  模型路径: {model_path}")
    print(f"  乘客文件: {passenger_file}")
    print("=" * 120)

    if not os.path.exists(passenger_file):
        print(f"乘客文件不存在: {passenger_file}")
        return
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"加载乘客: {passenger_file}")
    passenger_data = load_passengers_from_csv(passenger_file)
    n_passengers = len(passenger_data)
    print(f"   共 {n_passengers} 个乘客\n")

    print("加载模型...")
    device = torch.device("cpu")
    model = PPO.load(model_path, device=device)
    print(f"  模型加载成功 (设备: {device})\n")

    env = Vertiport8Env(max_steps=99999, mode="uniform")
    env.max_passengers = n_passengers

    all_passengers = []
    for p in passenger_data:
        origin = p['origin']
        dest = p['dest']
        action_a, info = get_shortest_action(origin, dest)

        walk_time_ticks = int(round(info['g1'] * 10))
        fly_time_ticks = int(round(info['fly'] * 10))
        final_walk_ticks = int(round(info['g2'] * 10))

        all_passengers.append({
            'id': p['id'],
            'origin': origin,
            'dest': dest,
            'action': action_a,
            'info': info,
            'walk_time_ticks': walk_time_ticks,
            'fly_time_ticks': fly_time_ticks,
            'final_walk_ticks': final_walk_ticks,
            'dep_vp': info['dep'],
            'arr_vp': info['arr']
        })

    print("决策信息 (前10个)")
    print("-" * 100)
    print(f"{'乘客':<6} {'起点':<20} {'终点':<20} {'决策':<25} {'地面1(分钟)':<12} {'理论(分钟)':<12}")
    print("-" * 100)
    for p in all_passengers[:10]:
        o = f"({p['origin'][0]:.1f},{p['origin'][1]:.1f})"
        d = f"({p['dest'][0]:.1f},{p['dest'][1]:.1f})"
        g1 = p['info']['g1'] if p['info']['g1'] > 0 else 0.0
        print(f"{p['id']:<6} {o:<20} {d:<20} {p['info']['type']:<25} {g1:<12.2f} {p['info']['total']:<12.2f}")
    print("-" * 100)

    env.pending_passengers = []
    for p in all_passengers:
        env.pending_passengers.append({
            'origin': p['origin'],
            'dest': p['dest'],
            'walk_time_ticks': p['walk_time_ticks'],
            'fly_time_ticks': p['fly_time_ticks'],
            'final_walk_ticks': p['final_walk_ticks'],
            'dep_vp': p['dep_vp'],
            'arr_vp': p['arr_vp']
        })

    env.step_count = 0
    env.passenger_id = 1
    env.total_spawned = 0
    env.passenger_origin = None
    env.passenger_dest = None
    env.walking_remaining = [-1] * n_passengers
    env.passenger_spawn_step = {}
    env.passenger_arrive_step = {}
    env.passenger_takeoff_step = {}
    env.passenger_final_step = {}
    env.queues = [[] for _ in range(8)]
    env.airport_aircraft = [[] for _ in range(8)]
    env.flying_aircraft = [[] for _ in range(8)]
    env.walking_passengers = []
    env.completed_passengers = []

    for i in range(env.num_aircraft):
        env.airport_aircraft[i % 8].append(i)

    env._load_next_passenger()

    step = 0
    completed_status = [0] * n_passengers

    print("\n仿真中...")

    action_counts = [0] * 7
    rank_names = ['最近', '第二近', '第三近', '第四近', '第五近', '第六近', '第七近']

    while step < max_ticks:
        all_takeoff = (
            len(env.completed_passengers) >= n_passengers and
            len(env.walking_passengers) == 0 and
            sum(len(q) for q in env.queues) == 0
        )
        if all_takeoff:
            break

        if env.passenger_origin is not None and env.passenger_dest is not None:
            origin = env.passenger_origin
            dest = env.passenger_dest
            pid = env.current_passenger_id

            obs = env._get_obs()

            action_rl, _ = model.predict(obs, deterministic=True)
            if isinstance(action_rl, np.ndarray):
                action_rl = int(action_rl.item()) if action_rl.size == 1 else int(action_rl[0])
            else:
                action_rl = int(action_rl)

            if 0 <= action_rl < 7:
                action_counts[action_rl] += 1

            _, info = get_shortest_action(origin, dest)

            if info['is_ground']:
                action_a = 56
                decision_str = f"地面 (RL)"
            else:
                arr_vp = info['arr']
                candidates = []
                for vp in range(8):
                    walk = distance_km(origin, VERTIPORTS[vp]) / WALK_SPEED * 10
                    fly = distance_km(VERTIPORTS[vp], VERTIPORTS[arr_vp]) / FLIGHT_SPEED * 10
                    total = walk + fly
                    candidates.append((vp, walk, fly, total))
                candidates.sort(key=lambda x: x[3])

                dep_vp = candidates[action_rl][0]
                action_a = get_action(dep_vp, arr_vp)

                rank_name = rank_names[action_rl] if action_rl < len(rank_names) else f"第{action_rl+1}近"
                decision_str = f"飞机({dep_vp}->{arr_vp}) [RL选择: {rank_name}]"
        else:
            action_a = 56
            pid = '--'
            decision_str = "无乘客"

        env.step(action_a)
        # 取消注释下面4行可打印每个tick的队列、空闲飞机、飞行中飞机
        # Q = [len(env.queues[vp]) for vp in range(8)]
        # A = [len(env.airport_aircraft[vp]) for vp in range(8)]
        # F = [len(env.flying_aircraft[vp]) for vp in range(8)]
        # print(f"tick={step:4d}  Q={Q}  A={A}  F={F}")
        step += 1

        if step % 500 == 0:
            for i, p in enumerate(all_passengers):
                pid_check = p['id']
                if p['info']['is_ground']:
                    if pid_check in env.passenger_final_step:
                        completed_status[i] = 1
                else:
                    if pid_check in env.passenger_takeoff_step:
                        completed_status[i] = 1

            completed_count = len(env.completed_passengers)
            print(f"  {step} ticks, 已完成 {completed_count}/{n_passengers} 人 ({completed_count / n_passengers * 100:.1f}%)")

            if all(s == 1 for s in completed_status):
                print(f"  所有乘客已完成！提前结束")
                break

    if step >= max_ticks:
        print(f"达到最大步数 {max_ticks} ticks ({max_ticks / 10:.0f} 分钟)，强制结束")
        print(f"   已完成 {len(env.completed_passengers)}/{n_passengers} 人")
        print(f"   走路中 {len(env.walking_passengers)} 人")
        print(f"   队列中 {sum(len(q) for q in env.queues)} 人")
        print(f"   飞行中 {sum(len(f) for f in env.flying_aircraft)} 架")

    print(f"\n仿真结束! 总步数: {step} ticks = {step / 10:.1f} 分钟\n")

    print("\n" + "=" * 80)
    print("统计信息 (PPO-12维)")
    print("=" * 80)

    completed_count = len(env.completed_passengers)
    print(f"  总乘客: {n_passengers}")
    print(f"  已完成: {completed_count}")
    print(f"  完成率: {completed_count / n_passengers * 100:.1f}%")
    print(f"  总步数: {step} ticks = {step / 10:.1f} 分钟")

    if completed_count > 0:
        all_ground = []
        all_wait = []
        all_flight = []
        all_total = []
        ground_completed = 0
        flight_completed = 0

        for cp in env.completed_passengers:
            walk = cp.get('walk_time', 0)
            wait = cp.get('wait_time', 0)
            fly = cp.get('fly_time', 0)
            final_walk = cp.get('final_walk', 0)
            total = cp.get('total_time', 0)

            all_ground.append(walk + final_walk)
            all_wait.append(wait)
            all_flight.append(fly)
            all_total.append(total)

            if fly == 0:
                ground_completed += 1
            else:
                flight_completed += 1

        print(f"\n已完成乘客分类:")
        print(f"  地面完成: {ground_completed}人")
        print(f"  飞行完成: {flight_completed}人")

        print(f"\n时间统计 (分钟)")
        print("-" * 40)
        print(f"  AGT (地面平均):   {np.mean(all_ground) / 10:.2f} 分钟")
        print(f"  AWT (等待平均):   {np.mean(all_wait) / 10:.2f} 分钟")
        print(f"  AAT (飞行平均):   {np.mean(all_flight) / 10:.2f} 分钟")
        print(f"  ATT (平均总时间): {np.mean(all_total) / 10:.2f} 分钟")
        print(
            f"  (AGT+AWT+AAT = {np.mean(all_ground) / 10 + np.mean(all_wait) / 10 + np.mean(all_flight) / 10:.2f} 分钟)")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/seven_choice_ppo/final_model.zip',
                        help='模型路径')
    parser.add_argument('--n_passengers', type=int, default=800)
    parser.add_argument('--passenger_file', type=str, default='data/passengers_800_test.csv')
    args = parser.parse_args()

    test_rl_model(args.model, args.passenger_file, args.n_passengers)