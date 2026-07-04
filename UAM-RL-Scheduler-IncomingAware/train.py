"""
train.py - 训练老PPO（12维，七选一）
"""

import os
import torch
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from seven_choice_env import SevenChoiceEnv


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


def make_env(passenger_file, seed=0, log_dir=None, max_steps=12000):
    def _init():
        env = SevenChoiceEnv(
            passenger_file=passenger_file,
            max_steps=max_steps,
            seed=seed
        )
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
        return env
    return _init


def train_seven_choice(
    total_timesteps=120_000,
    n_envs=1,
    passenger_file='data/passengers_800.csv',
    save_path='models/seven_choice_ppo',
    log_dir='./logs/seven_choice/',
    seed=42
):
    print("=" * 80)
    print("老PPO训练 (12维)")
    print("=" * 80)
    print(f"  总训练步数: {total_timesteps:,}")
    print(f"  并行环境数: {n_envs}")
    print(f"  模型保存: {save_path}")
    print("=" * 80)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)

    print("\n创建环境...")
    env = DummyVecEnv([
        make_env(passenger_file, seed + i, os.path.join(log_dir, f'env_{i}'))
        for i in range(n_envs)
    ])

    print("添加状态归一化...")
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    device = torch.device("cpu")
    print(f"使用设备: {device}")

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )

    print("\nPPO参数:")
    print(f"  学习率: 5e-4 (线性衰减)")
    print(f"  ent_coef: 0.028")
    print(f"  网络: 256×256")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(5e-4),
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.028,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(log_dir, 'tb')
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=save_path,
        name_prefix='seven_ppo',
        verbose=1
    )

    print("\n开始训练...")
    start_time = datetime.now()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True
    )

    elapsed = datetime.now() - start_time
    print(f"\n训练用时: {elapsed}")

    print("\n保存模型...")
    final_model_path = os.path.join(save_path, 'final_model')
    model.save(final_model_path)
    env.save(os.path.join(save_path, 'vec_normalize.pkl'))

    print(f"  模型: {final_model_path}.zip")
    print(f"  归一化: {save_path}/vec_normalize.pkl")
    env.close()

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"\n测试命令: python test.py --model {final_model_path}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=80_000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='models/seven_choice_ppo')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_seven_choice(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_path=args.save_path,
        seed=args.seed
    )