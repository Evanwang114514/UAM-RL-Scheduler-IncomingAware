"""
generate_passengers.py - 生成乘客数据
"""

import csv
import random
import os
from env import Vertiport8Env


def generate_passenger_file(n_passengers, filename, seed=42):
    random.seed(seed)
    env = Vertiport8Env(max_steps=99999, mode="uniform")
    env.max_passengers = n_passengers

    passengers = []
    for i in range(n_passengers):
        origin, dest = env._generate_one_passenger()
        passengers.append({
            'id': i + 1,
            'origin_x': origin[0],
            'origin_y': origin[1],
            'dest_x': dest[0],
            'dest_y': dest[1]
        })

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'origin_x', 'origin_y', 'dest_x', 'dest_y'])
        writer.writeheader()
        writer.writerows(passengers)

    print(f"已生成 {n_passengers} 个乘客 -> {filename}")
    return passengers


def load_passengers(filename):
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


def check_passenger_file(filename):
    passengers = load_passengers(filename)
    print(f"\n 检查 {filename}:")
    print(f"  总乘客: {len(passengers)}")
    print(f"  前5个乘客:")
    for p in passengers[:5]:
        print(
            f"    {p['id']}: origin({p['origin'][0]:.2f}, {p['origin'][1]:.2f}) -> dest({p['dest'][0]:.2f}, {p['dest'][1]:.2f})")
    return passengers


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # ========== 训练集 ==========
    generate_passenger_file(
        n_passengers=800,
        filename="data/passengers_800.csv",
        seed=42
    )

    # ========== 测试集（新随机种子） ==========
    generate_passenger_file(
        n_passengers=800,
        filename="data/passengers_800_test.csv",
        seed=123
    )

    # ========== 小规模测试 ==========
    generate_passenger_file(
        n_passengers=100,
        filename="data/passengers_100.csv",
        seed=42
    )

    # 检查
    check_passenger_file("data/passengers_100.csv")
    check_passenger_file("data/passengers_800.csv")
    check_passenger_file("data/passengers_800_test.csv")

    print("\n乘客文件生成完成!")
    print("   训练集: data/passengers_800.csv (seed=42)")
    print("   测试集: data/passengers_800_test.csv (seed=123)")