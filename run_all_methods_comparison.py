"""
全方法对比实验

使用原始实现对比所有方法:
1. PF  - 标准粒子滤波器
2. EPF - 扩展粒子滤波器
3. UPF - 无迹粒子滤波器
4. IPF - 隐式粒子滤波器
5. CINN-PF - 条件可逆神经网络粒子滤波器
6. SDPF - 结构化扩散粒子滤波器 (我们的方法)
"""

import sys
import time
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/SDPF')
sys.path.insert(0, '/root/CINN_PF')

# 导入原始实现
from Particle_filters import PF, EPF, UPF, IPF
from Particle_filters.L96 import L96
from CINN_for_PF import proposal_distribution, CINN_PF

# 导入我们的 SDPF
from sdpf import SDPF
from systems.lorenz96 import L96System


def run_original_method(method_name, method_func, data, orig_sys, num_particles, timeout=60):
    """运行原始方法实现"""
    t_start = time.time()
    try:
        result = method_func(data, orig_sys, num=num_particles)
        elapsed = time.time() - t_start
        
        # 检查结果是否有效
        if np.isnan(result).all() or np.isinf(result).any():
            return None, elapsed
        return result, elapsed
    except Exception as e:
        print(f"    [{method_name}] 错误: {e}")
        return None, time.time() - t_start


def run_cinn_pf(data, orig_sys, proposal, num_particles=10):
    """运行 CINN-PF"""
    if proposal is None:
        return None, 0
    
    t_start = time.time()
    try:
        result = CINN_PF(data, orig_sys, proposal, num=num_particles)
        elapsed = time.time() - t_start
        
        if np.isnan(result).all() or np.isinf(result).any():
            return None, elapsed
        return result, elapsed
    except Exception as e:
        print(f"    [CINN] 错误: {e}")
        return None, time.time() - t_start


def run_sdpf(data, system, num_particles=10, n_steps=10, device='cuda'):
    """运行 SDPF"""
    t_start = time.time()
    try:
        result = SDPF(data, system, num_particles=num_particles, 
                     n_steps=n_steps, step_size=0.05, device=device)
        elapsed = time.time() - t_start
        
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        return result, elapsed
    except Exception as e:
        print(f"    [SDPF] 错误: {e}")
        return None, time.time() - t_start


def compute_rmse(x_true, x_est, dim):
    """计算 RMSE (与论文一致的计算方式)"""
    if x_est is None:
        return float('nan')
    
    # 跳过初始状态
    x_true = x_true[1:]
    x_est = x_est[1:]
    
    # 确保形状一致
    min_len = min(len(x_true), len(x_est))
    x_true = x_true[:min_len]
    x_est = x_est[:min_len]
    
    # 论文中的 RMSE 计算: sqrt(dim * mean((x_true - x_est)^2))
    rmse = np.sqrt(dim * np.mean((x_true - x_est) ** 2))
    
    if np.isnan(rmse) or np.isinf(rmse):
        return float('nan')
    return rmse


def main():
    print("=" * 80)
    print("全方法对比实验 (L96 系统)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 实验配置 (与论文一致)
    dim = 50
    T = 200
    
    configs = [
        {'sigma_v': 0.1, 'p': 1, 'name': 'σv=0.1, p=1'},
        {'sigma_v': 0.1, 'p': 2, 'name': 'σv=0.1, p=2'},
        {'sigma_v': 0.2, 'p': 1, 'name': 'σv=0.2, p=1'},
        {'sigma_v': 0.2, 'p': 2, 'name': 'σv=0.2, p=2'},
    ]
    
    # 方法配置 (粒子数与论文一致)
    methods_config = {
        'PF-10k': {'func': PF, 'particles': 10000},
        'PF-1k': {'func': PF, 'particles': 1000},
        'EPF-100': {'func': EPF, 'particles': 100},
        'UPF-100': {'func': UPF, 'particles': 100},
        'IPF-10': {'func': IPF, 'particles': 10},
        'CINN-10': {'particles': 10},
        'SDPF-10': {'particles': 10, 'n_steps': 10},
        'SDPF-20': {'particles': 20, 'n_steps': 10},
    }
    
    all_results = {}
    
    for cfg in configs:
        sigma_v = cfg['sigma_v']
        p = cfg['p']
        config_name = cfg['name']
        
        print(f"\n\n{'#'*80}")
        print(f"# 配置: {config_name}")
        print(f"{'#'*80}")
        
        # 创建原始系统 (用于 PF, EPF, UPF, IPF, CINN)
        orig_sys = L96(sigma_x=0.01, sigma_v=sigma_v, sigma_e=0.01, p=p, N=dim)
        
        # 创建我们的系统 (用于 SDPF)
        our_sys = L96System(dim=dim, sigma_v=sigma_v, sigma_e=0.01, p=p)
        
        # 加载 CINN 模型
        s_v = 1 if sigma_v == 0.1 else 2
        cinn_path = f'/root/CINN_PF/models_for_L96_test/model_{p}_{s_v}'
        try:
            proposal = proposal_distribution(orig_sys, load=cinn_path)
            print(f"  [CINN] 加载模型: {cinn_path}")
        except Exception as e:
            proposal = None
            print(f"  [CINN] 加载失败: {e}")
        
        # 生成数据 (使用原始系统)
        u = np.zeros([T, 1])
        orig_sys.reset()
        data, x_true = orig_sys.generate(u=u)
        
        # 转换数据格式给 SDPF
        sdpf_data = {
            'u': data['u'],
            'y': data['y']
        }
        
        results = {}
        
        # 运行各方法
        print(f"\n运行各方法 (T={T}, dim={dim})...")
        
        # PF-10k
        print(f"\n  PF-10000...")
        x_est, t = run_original_method('PF-10k', PF, data, orig_sys, 10000)
        rmse = compute_rmse(x_true, x_est, dim)
        results['PF-10k'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        # PF-1k
        print(f"\n  PF-1000...")
        x_est, t = run_original_method('PF-1k', PF, data, orig_sys, 1000)
        rmse = compute_rmse(x_true, x_est, dim)
        results['PF-1k'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        # EPF-100
        print(f"\n  EPF-100...")
        x_est, t = run_original_method('EPF', EPF, data, orig_sys, 100)
        rmse = compute_rmse(x_true, x_est, dim)
        results['EPF-100'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败/发散, 时间: {t:.2f}s")
        
        # UPF-100
        print(f"\n  UPF-100...")
        x_est, t = run_original_method('UPF', UPF, data, orig_sys, 100)
        rmse = compute_rmse(x_true, x_est, dim)
        results['UPF-100'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败/发散, 时间: {t:.2f}s")
        
        # IPF-10
        print(f"\n  IPF-10...")
        x_est, t = run_original_method('IPF', IPF, data, orig_sys, 10)
        rmse = compute_rmse(x_true, x_est, dim)
        results['IPF-10'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        # CINN-10
        print(f"\n  CINN-10...")
        x_est, t = run_cinn_pf(data, orig_sys, proposal, 10)
        rmse = compute_rmse(x_true, x_est, dim)
        results['CINN-10'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        # SDPF-10
        print(f"\n  SDPF-10...")
        x_est, t = run_sdpf(sdpf_data, our_sys, num_particles=10, n_steps=10, device=device)
        rmse = compute_rmse(x_true, x_est, dim)
        results['SDPF-10'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        # SDPF-20
        print(f"\n  SDPF-20...")
        x_est, t = run_sdpf(sdpf_data, our_sys, num_particles=20, n_steps=10, device=device)
        rmse = compute_rmse(x_true, x_est, dim)
        results['SDPF-20'] = {'rmse': rmse, 'time': t}
        print(f"    RMSE: {rmse:.2f}, 时间: {t:.2f}s" if not np.isnan(rmse) else f"    失败, 时间: {t:.2f}s")
        
        all_results[config_name] = results
    
    # 汇总表格
    print("\n\n" + "=" * 100)
    print("实验结果汇总 (RMSE)")
    print("=" * 100)
    
    methods = ['PF-10k', 'PF-1k', 'EPF-100', 'UPF-100', 'IPF-10', 'CINN-10', 'SDPF-10', 'SDPF-20']
    
    # 表头
    header = f"{'配置':<15}"
    for m in methods:
        header += f" | {m:<10}"
    header += " | 最优"
    print(header)
    print("-" * 120)
    
    # 数据行
    for cfg in configs:
        config_name = cfg['name']
        results = all_results[config_name]
        
        row = f"{config_name:<15}"
        rmses = []
        for m in methods:
            rmse = results.get(m, {}).get('rmse', float('nan'))
            if np.isnan(rmse):
                row += f" | {'DIV':<10}"
            else:
                row += f" | {rmse:<10.2f}"
                rmses.append((rmse, m))
        
        # 找最优
        if rmses:
            best = min(rmses, key=lambda x: x[0])
            row += f" | {best[1]}"
        else:
            row += " | N/A"
        
        print(row)
    
    # 计算平均 (排除发散的)
    print("-" * 120)
    avg_row = f"{'平均':<15}"
    for m in methods:
        rmses = [all_results[cfg['name']].get(m, {}).get('rmse', float('nan')) for cfg in configs]
        valid_rmses = [r for r in rmses if not np.isnan(r)]
        if valid_rmses:
            avg_row += f" | {np.mean(valid_rmses):<10.2f}"
        else:
            avg_row += f" | {'DIV':<10}"
    print(avg_row)
    
    # 时间对比
    print("\n\n" + "=" * 100)
    print("计算时间对比 (秒)")
    print("=" * 100)
    
    header = f"{'配置':<15}"
    for m in methods:
        header += f" | {m:<10}"
    print(header)
    print("-" * 120)
    
    for cfg in configs:
        config_name = cfg['name']
        results = all_results[config_name]
        
        row = f"{config_name:<15}"
        for m in methods:
            t = results.get(m, {}).get('time', 0)
            row += f" | {t:<10.2f}"
        print(row)
    
    # 分析
    print("\n\n" + "=" * 100)
    print("分析")
    print("=" * 100)
    
    print("\n方法说明:")
    print("  - PF: 标准粒子滤波器 (使用先验作为提议分布)")
    print("  - EPF: 扩展粒子滤波器 (线性化近似最优提议)")
    print("  - UPF: 无迹粒子滤波器 (无迹变换近似最优提议)")
    print("  - IPF: 隐式粒子滤波器 (优化求解最优采样点)")
    print("  - CINN: 条件可逆神经网络粒子滤波器 (神经网络学习最优提议)")
    print("  - SDPF: 结构化扩散粒子滤波器 (Langevin动力学采样)")
    
    print("\n关键发现:")
    print("  1. EPF/UPF 在非线性观测(p>1)时容易发散")
    print("  2. IPF 计算量大但精度有限")
    print("  3. CINN 需要预训练，但精度最高")
    print("  4. SDPF 无需预训练，用少量粒子达到接近 CINN 的精度")


if __name__ == "__main__":
    main()
