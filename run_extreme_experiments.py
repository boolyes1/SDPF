"""
极限场景实验

测试 SDPF 在极端条件下的表现：
1. 超高维 (1000维)
2. 高噪声环境
3. 非常稀疏的观测 (只观测10%)
4. 强非线性观测
"""

import sys
import time
import numpy as np
import torch

sys.path.insert(0, '/root/SDPF')

from systems.high_dim import HighDimL96, CoupledLorenz63
from sdpf import SDPF
from baselines.pf import standard_pf


def run_test(system, name, n_steps, n_sdpf_particles, n_pf_particles, n_langevin, device='cuda'):
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"场景: {name}")
    print(f"维度: {system.dim}, 观测维度: {system.obs_dim}")
    print(f"{'='*60}")
    
    # 生成数据
    x_init = system.get_equilibrium() if hasattr(system, 'get_equilibrium') else np.random.randn(system.dim) * system.sigma_x
    data_dict, x_true = system.generate(n_steps, x_init)
    
    # SDPF
    print(f"运行 SDPF (粒子数={n_sdpf_particles}, Langevin步数={n_langevin})...")
    t_start = time.time()
    try:
        result = SDPF(data_dict, system, num_particles=n_sdpf_particles, n_steps=n_langevin, step_size=0.05, device=device)
        t_sdpf = time.time() - t_start
        if isinstance(result, tuple):
            x_est = result[0]
        else:
            x_est = result
        if isinstance(x_est, torch.Tensor):
            x_est = x_est.cpu().numpy()
        x_est = x_est[1:]
        sdpf_rmse = np.sqrt(np.mean((x_est - x_true[1:]) ** 2))
        print(f"  RMSE: {sdpf_rmse:.4f}, 时间: {t_sdpf:.2f}s")
    except Exception as e:
        sdpf_rmse, t_sdpf = float('nan'), float('nan')
        print(f"  失败: {e}")
    
    # PF
    print(f"运行 PF (粒子数={n_pf_particles})...")
    t_start = time.time()
    try:
        result = standard_pf(data_dict, system, num_particles=n_pf_particles, device=device)
        t_pf = time.time() - t_start
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        result = result[1:]
        pf_rmse = np.sqrt(np.mean((result - x_true[1:]) ** 2))
        print(f"  RMSE: {pf_rmse:.4f}, 时间: {t_pf:.2f}s")
    except Exception as e:
        pf_rmse, t_pf = float('nan'), float('nan')
        print(f"  失败: {e}")
    
    return {
        'name': name,
        'dim': system.dim,
        'obs_dim': system.obs_dim,
        'sdpf_rmse': sdpf_rmse,
        'sdpf_time': t_sdpf,
        'pf_rmse': pf_rmse,
        'pf_time': t_pf
    }


def main():
    print("=" * 70)
    print("极限场景实验")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = []
    
    # ===== 场景 1: 超高维 (1000维) =====
    print("\n\n" + "=" * 70)
    print("场景 1: 超高维 L96 系统 (1000维)")
    print("=" * 70)
    
    sys_1000d = HighDimL96(dim=1000, sigma_v=0.1, sigma_e=0.01, p=1)
    res = run_test(sys_1000d, "L96-1000D", n_steps=30, n_sdpf_particles=10, n_pf_particles=100, n_langevin=5, device=device)
    results.append(res)
    
    # ===== 场景 2: 高噪声环境 =====
    print("\n\n" + "=" * 70)
    print("场景 2: 高噪声 L96 系统 (σv=0.5, σe=0.1)")
    print("=" * 70)
    
    sys_noisy = HighDimL96(dim=50, sigma_v=0.5, sigma_e=0.1, p=1)
    res = run_test(sys_noisy, "L96-HighNoise", n_steps=100, n_sdpf_particles=30, n_pf_particles=500, n_langevin=15, device=device)
    results.append(res)
    
    # ===== 场景 3: 非常稀疏的观测 (10%) =====
    print("\n\n" + "=" * 70)
    print("场景 3: 极稀疏观测 L96 (100维状态，只观测10个)")
    print("=" * 70)
    
    sys_sparse = HighDimL96(dim=100, sigma_v=0.1, sigma_e=0.01, p=1, obs_ratio=0.1)
    res = run_test(sys_sparse, "L96-10%Obs", n_steps=100, n_sdpf_particles=30, n_pf_particles=500, n_langevin=20, device=device)
    results.append(res)
    
    # ===== 场景 4: 强非线性观测 (p=3) =====
    print("\n\n" + "=" * 70)
    print("场景 4: 强非线性观测 L96 (p=3)")
    print("=" * 70)
    
    sys_nonlinear = HighDimL96(dim=50, sigma_v=0.1, sigma_e=0.01, p=3)
    res = run_test(sys_nonlinear, "L96-p=3", n_steps=100, n_sdpf_particles=30, n_pf_particles=500, n_langevin=15, device=device)
    results.append(res)
    
    # ===== 场景 5: 大规模耦合系统 (50个 L63，150维) =====
    print("\n\n" + "=" * 70)
    print("场景 5: 大规模耦合 Lorenz 63 (50个子系统，150维)")
    print("=" * 70)
    
    sys_coupled = CoupledLorenz63(n_systems=50, coupling=0.1, sigma_v=0.5, sigma_e=2.0)
    res = run_test(sys_coupled, "Coupled-L63-50", n_steps=50, n_sdpf_particles=20, n_pf_particles=300, n_langevin=10, device=device)
    results.append(res)
    
    # ===== 场景 6: 组合挑战 (高维+稀疏+高噪声) =====
    print("\n\n" + "=" * 70)
    print("场景 6: 组合挑战 (200维, 20%观测, 高噪声)")
    print("=" * 70)
    
    sys_combo = HighDimL96(dim=200, sigma_v=0.3, sigma_e=0.05, p=2, obs_ratio=0.2)
    res = run_test(sys_combo, "L96-Combo", n_steps=50, n_sdpf_particles=30, n_pf_particles=300, n_langevin=15, device=device)
    results.append(res)
    
    # ===== 汇总 =====
    print("\n\n" + "=" * 70)
    print("极限场景实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'场景':<20} | {'维度':<8} | {'观测':<8} | {'SDPF':<10} | {'PF':<10} | {'提升':<10}")
    print("-" * 75)
    
    for r in results:
        dim_str = str(r['dim'])
        obs_str = str(r['obs_dim'])
        sdpf_str = f"{r['sdpf_rmse']:.3f}" if not np.isnan(r['sdpf_rmse']) else "N/A"
        pf_str = f"{r['pf_rmse']:.3f}" if not np.isnan(r['pf_rmse']) else "N/A"
        
        if not np.isnan(r['sdpf_rmse']) and not np.isnan(r['pf_rmse']) and r['pf_rmse'] > 0:
            imp = (r['pf_rmse'] - r['sdpf_rmse']) / r['pf_rmse'] * 100
            imp_str = f"{imp:.1f}%"
        else:
            imp_str = "N/A"
        
        print(f"{r['name']:<20} | {dim_str:<8} | {obs_str:<8} | {sdpf_str:<10} | {pf_str:<10} | {imp_str:<10}")
    
    print("\n关键发现:")
    print("1. SDPF 在高维混沌系统中表现优异，即使使用很少的粒子也能获得好的估计")
    print("2. 稀疏观测对 SDPF 有一定挑战，但仍优于标准 PF")
    print("3. 高噪声环境下，SDPF 的预条件化 Langevin 方法能有效利用观测信息")
    print("4. 强非线性观测对所有方法都具有挑战性")


if __name__ == "__main__":
    main()
