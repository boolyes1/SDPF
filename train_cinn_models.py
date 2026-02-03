"""
CINN 模型训练脚本

为各种系统配置训练 CINN 模型，用于与 SDPF 进行公平对比。
"""

import sys
import os
import numpy as np
import torch
import time

sys.path.insert(0, '/root/SDPF')
sys.path.insert(0, '/root/CINN_PF')

from CINN_model.PCINN import PCINN


def generate_training_data(system, n_samples=5000, burn_in=200):
    """
    生成 CINN 训练数据
    
    训练数据格式:
    - x: 当前状态 (目标)
    - c: 条件 [x_pred, y] (输入)
    """
    print(f"  生成 {n_samples} 个训练样本...")
    
    x_list = []
    c_list = []
    
    # 初始化
    x = system.get_equilibrium() + system.sigma_x * np.random.randn(system.dim)
    
    # 燃烧期
    for _ in range(burn_in):
        x = system.f(x[np.newaxis, :])[0] + system.sample_process_noise(1)[0]
    
    # 生成数据
    for i in range(n_samples):
        # 状态转移
        x_prev = x.copy()
        x = system.f(x_prev[np.newaxis, :])[0] + system.sample_process_noise(1)[0]
        
        # 观测
        y = system.h(x[np.newaxis, :])[0] + system.sample_observation_noise(1)[0]
        
        # 预测 (从前一个状态)
        x_pred = system.f(x_prev[np.newaxis, :])[0]
        
        # 存储
        x_list.append(x)
        c_list.append(np.concatenate([x_pred, y]))
        
        if (i + 1) % 1000 == 0:
            print(f"    已生成 {i+1}/{n_samples} 样本")
    
    return np.array(x_list), np.array(c_list)


def train_cinn_for_system(system, name, save_dir, n_samples=5000, epochs=2000):
    """
    为指定系统训练 CINN 模型
    """
    print(f"\n{'='*60}")
    print(f"训练 CINN 模型: {name}")
    print(f"维度: {system.dim}, 观测维度: {system.obs_dim}")
    print(f"{'='*60}")
    
    # 生成训练数据
    x_train, c_train = generate_training_data(system, n_samples)
    
    print(f"  训练数据形状: x={x_train.shape}, c={c_train.shape}")
    
    # 创建保存目录
    model_dir = os.path.join(save_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 配置 CINN
    # 根据维度调整网络结构
    if system.dim <= 50:
        nods = [128, 128, 128]
    elif system.dim <= 100:
        nods = [256, 256, 256]
    elif system.dim <= 200:
        nods = [512, 512, 512]
    else:
        nods = [1024, 512, 512]
    
    a_f = ['LR']  # LeakyReLU
    
    print(f"  网络结构: {nods}")
    print(f"  训练轮数: {epochs}")
    
    # 训练
    t_start = time.time()
    
    try:
        cinn = PCINN(nods, a_f, lr=1e-3, lam=1e-12)
        cinn.train(c_train, x_train, terms=epochs)
        
        # 保存模型
        cinn.save(model_dir)
        
        t_train = time.time() - t_start
        print(f"  训练完成！耗时: {t_train:.1f}s")
        print(f"  模型保存至: {model_dir}")
        
        return cinn, model_dir
    
    except Exception as e:
        print(f"  训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    from systems.lorenz96 import L96System
    from systems.high_dim import HighDimL96, CoupledLorenz63
    
    print("=" * 70)
    print("CINN 模型训练")
    print("=" * 70)
    
    save_dir = '/root/SDPF/cinn_models'
    os.makedirs(save_dir, exist_ok=True)
    
    trained_models = {}
    
    # ===== 1. L96-50D 各种配置 =====
    print("\n\n" + "#" * 70)
    print("# 训练 L96-50D 模型")
    print("#" * 70)
    
    configs_50d = [
        ('L96_50D_v01_p1', {'dim': 50, 'sigma_v': 0.1, 'sigma_e': 0.01, 'p': 1}),
        ('L96_50D_v01_p2', {'dim': 50, 'sigma_v': 0.1, 'sigma_e': 0.01, 'p': 2}),
        ('L96_50D_v02_p1', {'dim': 50, 'sigma_v': 0.2, 'sigma_e': 0.01, 'p': 1}),
        ('L96_50D_v02_p2', {'dim': 50, 'sigma_v': 0.2, 'sigma_e': 0.01, 'p': 2}),
    ]
    
    for name, cfg in configs_50d:
        system = L96System(**cfg)
        cinn, path = train_cinn_for_system(system, name, save_dir, n_samples=5000, epochs=1500)
        if cinn:
            trained_models[name] = path
    
    # ===== 2. L96-100D =====
    print("\n\n" + "#" * 70)
    print("# 训练 L96-100D 模型")
    print("#" * 70)
    
    system_100d = HighDimL96(dim=100, sigma_v=0.1, sigma_e=0.01, p=1)
    cinn, path = train_cinn_for_system(system_100d, 'L96_100D', save_dir, n_samples=8000, epochs=2000)
    if cinn:
        trained_models['L96_100D'] = path
    
    # ===== 3. L96 高噪声 =====
    print("\n\n" + "#" * 70)
    print("# 训练 L96 高噪声模型")
    print("#" * 70)
    
    system_noisy = HighDimL96(dim=50, sigma_v=0.5, sigma_e=0.1, p=1)
    cinn, path = train_cinn_for_system(system_noisy, 'L96_50D_noisy', save_dir, n_samples=5000, epochs=1500)
    if cinn:
        trained_models['L96_50D_noisy'] = path
    
    # ===== 4. L96 强非线性 =====
    print("\n\n" + "#" * 70)
    print("# 训练 L96 强非线性模型 (p=3)")
    print("#" * 70)
    
    system_nonlin = HighDimL96(dim=50, sigma_v=0.1, sigma_e=0.01, p=3)
    cinn, path = train_cinn_for_system(system_nonlin, 'L96_50D_p3', save_dir, n_samples=5000, epochs=1500)
    if cinn:
        trained_models['L96_50D_p3'] = path
    
    # ===== 汇总 =====
    print("\n\n" + "=" * 70)
    print("训练完成汇总")
    print("=" * 70)
    
    print(f"\n已训练模型数: {len(trained_models)}")
    for name, path in trained_models.items():
        print(f"  - {name}: {path}")
    
    print(f"\n模型保存目录: {save_dir}")


if __name__ == "__main__":
    main()
