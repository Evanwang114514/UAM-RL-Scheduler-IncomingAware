"""
test_all_ground.py - 全部走地面测试
所有乘客强制选择地面，不坐飞机
"""

import numpy as np
import csv
import os
import argparse
from env import Vertiport8Env

CAV_SPEED_KPH = 20.0
WALK_SPEED = CAV_SPEED_KPH / 60.0


def distance_km(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


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


def test_all_ground(passenger_file='data/passengers_800_test.csv', n_passengers=800, max_ticks=12000):
    print("=" * 120)
    print(f"全部走地面测试 - {n_passengers}人 (测试集)")
    print("=" * 120)
    print(f"  地面速度: {CAV_SPEED_KPH} km/h = {WALK_SPEED:.3f} km/min")
    print(f"  乘客文件: {passenger_file}")
    print("=" * 120)

    if not os.path.exists(passenger_file):
        print(f"乘客文件不存在: {passenger_file}")
        return

    print(f"加载乘客: {passenger_file}")
    passenger_data = load_passengers_from_csv(passenger_file)
    n_passengers = len(passenger_data)
    print(f"   共 {n_passengers} 个乘客\n")

    env = Vertiport8Env(max_steps=99999, mode="uniform")
    env.max_passengers = n_passengers

    all_passengers = []
    for p in passenger_data:
        origin = p['origin']
        dest = p['dest']
        dist = distance_km(origin, dest)
        walk_time_minutes = dist / WALK_SPEED
        walk_time_ticks = int(round(walk_time_minutes * 10))

        all_passengers.append({
            'id': p['id'],
            'origin': origin,
            'dest': dest,
            'walk_time_ticks': walk_time_ticks,
            'walk_time_minutes': walk_time_minutes,
            'distance_km': dist
        })

    print("决策信息 (前10个)")
    print("-" * 100)
    print(f"{'乘客':<6} {'起点':<20} {'终点':<20} {'距离(km)':<12} {'走路(分钟)':<12}")
    print("-" * 100)
    for p in all_passengers[:10]:
        o = f"({p['origin'][0]:.1f},{p['origin'][1]:.1f})"
        d = f"({p['dest'][0]:.1f},{p['dest'][1]:.1f})"
        print(f"{p['id']:<6} {o:<20} {d:<20} {p['distance_km']:<12.2f} {p['walk_time_minutes']:<12.2f}")
    print("-" * 100)

    env.pending_passengers = []
    for p in all_passengers:
        env.pending_passengers.append({
            'origin': p['origin'],
            'dest': p['dest'],
            'walk_time_ticks': p['walk_time_ticks'],
            'fly_time_ticks': 0,
            'final_walk_ticks': 0,
            'dep_vp': -1,
            'arr_vp': -1
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
    print("\n仿真中...")

    while step < max_ticks:
        all_done = (
            len(env.completed_passengers) >= n_passengers and
            len(env.walking_passengers) == 0 and
            sum(len(q) for q in env.queues) == 0
        )
        if all_done:
            break

        action_a = 56
        env.step(action_a)
        step += 1

        if step % 500 == 0:
            completed_count = len(env.completed_passengers)
            print(f"  {step} ticks, 已完成 {completed_count}/{n_passengers} 人 ({completed_count / n_passengers * 100:.1f}%)")

    print(f"\n仿真结束! 总步数: {step} ticks = {step / 10:.1f} 分钟\n")

    print("\n" + "=" * 80)
    print("统计信息 (全部走地面 - 测试集)")
    print("=" * 80)

    completed_count = len(env.completed_passengers)
    print(f"  总乘客: {n_passengers}")
    print(f"  已完成: {completed_count}")
    print(f"  完成率: {completed_count / n_passengers * 100:.1f}%")
    print(f"  总步数: {step} ticks = {step / 10:.1f} 分钟")

    if completed_count > 0:
        all_ground = []
        all_total = []
        distances = []

        for cp in env.completed_passengers:
            walk = cp.get('walk_time', 0)
            total = cp.get('total_time', 0)
            all_ground.append(walk)
            all_total.append(total)

        for p in all_passengers:
            distances.append(p['distance_km'])

        print(f"\n时间统计 (分钟)")
        print("-" * 40)
        print(f"  平均地面距离:    {np.mean(distances):.2f} km")
        print(f"  平均地面时间:    {np.mean(all_ground) / 10:.2f} 分钟")
        print(f"  平均总时间:      {np.mean(all_total) / 10:.2f} 分钟")
        print(f"  最大总时间:      {np.max(all_total) / 10:.2f} 分钟")
        print(f"  最小总时间:      {np.min(all_total) / 10:.2f} 分钟")
        print(f"  总时间标准差:    {np.std(all_total) / 10:.2f} 分钟")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_passengers', type=int, default=800)
    parser.add_argument('--passenger_file', type=str, default='data/passengers_800_test.csv')
    args = parser.parse_args()

    test_all_ground(args.passenger_file, args.n_passengers)