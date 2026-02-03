"""
高维场景实验

测试 SDPF 在多种高维状态估计场景中的表现：
1. 高维 L96 (100维、200维)
2. 稀疏观测 L96 (50维状态，只观测25维)
3. 耦合 Lorenz 63 (10个系统耦合，30维)
4. 高维线性系统 (100维状态，50维观测)
"""

import sys
import time
import numpy as np
import torch

sys.path.insert(0, '/root/SDPF')

from systems.high_dim import HighDimL96, CoupledLorenz63, HighDimLinear
from sdpf import SDPF


def run_experiment(system, name, n_steps=100, n_particles=20, n_langevin=10, device='cuda'):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"系统: {name}")
    print(f"维度: {system.dim}, 观测维度: {system.obs_dim}")
    print(f"{'='*60}")
    
    # 生成数据
    print("生成测试数据...")
    x_init = system.get_equilibrium() if hasattr(system, 'get_equilibrium') else np.random.randn(system.dim) * system.sigma_x
    data_dict, x_true = system.generate(n_steps, x_init)
    data_dict['x'] = x_true  # 添加真实状态以便 SDPF 计算
    
    # 运行 SDPF
    print(f"运行 SDPF (粒子数={n_particles}, Langevin步数={n_langevin})...")
    t_start = time.time()
    
    try:
        result = SDPF(
            data_dict, system, 
            num_particles=n_particles, 
            n_steps=n_langevin, 
            step_size=0.05,
            device=device
        )
        t_sdpf = time.time() - t_start
        
        # 处理返回值（可能是 tensor 或 tuple）
        if isinstance(result, tuple):
            x_est, weights = result
        else:
            x_est = result
        
        # 转换为 NumPy
        if isinstance(x_est, torch.Tensor):
            x_est = x_est.cpu().numpy()
        
        x_est = x_est[1:]  # 跳过初始状态
        
        # 计算 RMSE
        x_true_eval = x_true[1:, :]  # (T, dim)
        if x_est.shape[0] != x_true_eval.shape[0]:
            x_est = x_est[:x_true_eval.shape[0], :]
        
        rmse = np.sqrt(np.mean((x_est - x_true_eval) ** 2))
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  时间: {t_sdpf:.2f}s")
        
        return rmse, t_sdpf
    
    except Exception as e:
        import traceback
        print(f"  错误: {e}")
        traceback.print_exc()
        return float('nan'), float('nan')


def run_pf_baseline(system, name, n_steps=100, n_particles=1000, device='cuda'):
    """运行标准 PF 作为对比"""
    from baselines.pf import standard_pf
    
    print(f"\n运行标准 PF (粒子数={n_particles})...")
    
    # 生成数据 (使用与 SDPF 相同的数据)
    x_init = system.get_equilibrium() if hasattr(system, 'get_equilibrium') else np.random.randn(system.dim) * system.sigma_x
    data_dict, x_true = system.generate(n_steps, x_init)
    
    t_start = time.time()
    
    try:
        x_est = standard_pf(data_dict, system, num_particles=n_particles, device=device)
        t_pf = time.time() - t_start
        
        # 转换为 NumPy
        if isinstance(x_est, torch.Tensor):
            x_est = x_est.cpu().numpy()
        
        x_true_eval = x_true[1:, :]
        x_est_eval = x_est[1:]
        if x_est_eval.shape[0] != x_true_eval.shape[0]:
            x_est_eval = x_est_eval[:x_true_eval.shape[0], :]
        
        rmse = np.sqrt(np.mean((x_est_eval - x_true_eval) ** 2))
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  时间: {t_pf:.2f}s")
        
        return rmse, t_pf
    
    except Exception as e:
        import traceback
        print(f"  错误: {e}")
        traceback.print_exc()
        return float('nan'), float('nan')


def main():
    print("=" * 70)
    print("高维状态估计场景实验")
    print("=" * 70)
    
    # 检查 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    # ===== 场景 1: 高维 L96 (100维) =====
    print("\n\n" + "=" * 70)
    print("场景 1: 高维 L96 系统 (100维)")
    print("=" * 70)
    
    sys_l96_100 = HighDimL96(dim=100, sigma_v=0.1, sigma_e=0.01, p=1)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_l96_100, "L96-100D", 
        n_steps=100, n_particles=20, n_langevin=10, device=device
    )
    rmse_pf, t_pf = run_pf_baseline(
        sys_l96_100, "L96-100D", 
        n_steps=100, n_particles=500, device=device
    )
    results['L96-100D'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 场景 2: 高维 L96 (200维) =====
    print("\n\n" + "=" * 70)
    print("场景 2: 高维 L96 系统 (200维)")
    print("=" * 70)
    
    sys_l96_200 = HighDimL96(dim=200, sigma_v=0.1, sigma_e=0.01, p=1)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_l96_200, "L96-200D", 
        n_steps=50, n_particles=20, n_langevin=10, device=device
    )
    rmse_pf, t_pf = run_pf_baseline(
        sys_l96_200, "L96-200D", 
        n_steps=50, n_particles=500, device=device
    )
    results['L96-200D'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 场景 3: 稀疏观测 L96 (50维状态，只观测25个) =====
    print("\n\n" + "=" * 70)
    print("场景 3: 稀疏观测 L96 系统 (50维状态，25维观测)")
    print("=" * 70)
    
    sys_sparse = HighDimL96(dim=50, sigma_v=0.1, sigma_e=0.01, p=1, obs_ratio=0.5)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_sparse, "L96-Sparse", 
        n_steps=100, n_particles=20, n_langevin=15, device=device
    )
    rmse_pf, t_pf = run_pf_baseline(
        sys_sparse, "L96-Sparse", 
        n_steps=100, n_particles=500, device=device
    )
    results['L96-Sparse'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 场景 4: 耦合 Lorenz 63 (10个系统，30维) =====
    print("\n\n" + "=" * 70)
    print("场景 4: 耦合 Lorenz 63 系统 (10个子系统，30维)")
    print("=" * 70)
    
    sys_coupled = CoupledLorenz63(n_systems=10, coupling=0.1, sigma_v=0.5, sigma_e=2.0)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_coupled, "Coupled-L63", 
        n_steps=100, n_particles=20, n_langevin=10, device=device
    )
    rmse_pf, t_pf = run_pf_baseline(
        sys_coupled, "Coupled-L63", 
        n_steps=100, n_particles=500, device=device
    )
    results['Coupled-L63'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 场景 5: 高维线性系统 (100维状态，50维观测) =====
    print("\n\n" + "=" * 70)
    print("场景 5: 高维线性系统 (100维状态，50维观测)")
    print("=" * 70)
    
    sys_linear = HighDimLinear(dim=100, obs_dim=50, stability=0.95, sigma_v=0.1, sigma_e=0.1)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_linear, "Linear-100D", 
        n_steps=100, n_particles=20, n_langevin=10, device=device
    )
    rmse_pf, t_pf = run_pf_baseline(
        sys_linear, "Linear-100D", 
        n_steps=100, n_particles=500, device=device
    )
    results['Linear-100D'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 场景 6: 超高维 L96 (500维) =====
    print("\n\n" + "=" * 70)
    print("场景 6: 超高维 L96 系统 (500维) - 极限测试")
    print("=" * 70)
    
    sys_l96_500 = HighDimL96(dim=500, sigma_v=0.1, sigma_e=0.01, p=1)
    rmse_sdpf, t_sdpf = run_experiment(
        sys_l96_500, "L96-500D", 
        n_steps=30, n_particles=10, n_langevin=5, device=device
    )
    # 标准 PF 在 500 维可能效果很差，但还是测试一下
    rmse_pf, t_pf = run_pf_baseline(
        sys_l96_500, "L96-500D", 
        n_steps=30, n_particles=200, device=device
    )
    results['L96-500D'] = {
        'SDPF': (rmse_sdpf, t_sdpf),
        'PF': (rmse_pf, t_pf)
    }
    
    # ===== 汇总结果 =====
    print("\n\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'场景':<20} | {'SDPF RMSE':<12} | {'PF RMSE':<12} | {'SDPF 时间':<10} | {'PF 时间':<10} | {'提升比例':<10}")
    print("-" * 90)
    
    for name, res in results.items():
        sdpf_rmse, sdpf_t = res['SDPF']
        pf_rmse, pf_t = res['PF']
        
        if not np.isnan(sdpf_rmse) and not np.isnan(pf_rmse) and pf_rmse > 0:
            improvement = (pf_rmse - sdpf_rmse) / pf_rmse * 100
            imp_str = f"{improvement:.1f}%"
        else:
            imp_str = "N/A"
        
        sdpf_rmse_str = f"{sdpf_rmse:.3f}" if not np.isnan(sdpf_rmse) else "失败"
        pf_rmse_str = f"{pf_rmse:.3f}" if not np.isnan(pf_rmse) else "失败"
        sdpf_t_str = f"{sdpf_t:.2f}s" if not np.isnan(sdpf_t) else "N/A"
        pf_t_str = f"{pf_t:.2f}s" if not np.isnan(pf_t) else "N/A"
        
        print(f"{name:<20} | {sdpf_rmse_str:<12} | {pf_rmse_str:<12} | {sdpf_t_str:<10} | {pf_t_str:<10} | {imp_str:<10}")
    
    print("\n说明:")
    print("- SDPF 使用 10-20 个粒子，标准 PF 使用 200-500 个粒子")
    print("- 提升比例 = (PF_RMSE - SDPF_RMSE) / PF_RMSE * 100%")
    print("- 正值表示 SDPF 优于 PF，负值表示 PF 优于 SDPF")
    
    # 维度扩展性分析
    print("\n\n" + "=" * 70)
    print("维度扩展性分析")
    print("=" * 70)
    
    dims = [50, 100, 200]
    print(f"\n{'维度':<10} | {'SDPF RMSE':<12} | {'SDPF 时间':<12}")
    print("-" * 40)
    
    for dim in dims:
        key = f"L96-{dim}D" if dim >= 100 else "L96-Sparse"
        if key in results:
            rmse, t = results[key]['SDPF']
            rmse_str = f"{rmse:.3f}" if not np.isnan(rmse) else "N/A"
            t_str = f"{t:.2f}s" if not np.isnan(t) else "N/A"
            print(f"{dim:<10} | {rmse_str:<12} | {t_str}")
    
    if 'L96-500D' in results:
        rmse, t = results['L96-500D']['SDPF']
        rmse_str = f"{rmse:.3f}" if not np.isnan(rmse) else "N/A"
        t_str = f"{t:.2f}s" if not np.isnan(t) else "N/A"
        print(f"{500:<10} | {rmse_str:<12} | {t_str}")


if __name__ == "__main__":
    main()
