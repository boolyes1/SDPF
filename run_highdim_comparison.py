"""
高维场景全方法对比实验

对比方法:
1. PF (标准粒子滤波器)
2. EPF (扩展粒子滤波器)
3. SDPF (结构化扩散粒子滤波器)
4. CINN-PF (需要预训练，仅在 L96-50D 上可用)

测试场景:
- L96-50D (基准)
- L96-100D
- L96-200D
- L96-500D
- 稀疏观测
- 高噪声
- 强非线性
"""

import sys
import time
import numpy as np
import torch

sys.path.insert(0, '/root/SDPF')
sys.path.insert(0, '/root/CINN_PF')

from systems.high_dim import HighDimL96, CoupledLorenz63
from systems.lorenz96 import L96System
from sdpf import SDPF
from baselines.pf import standard_pf
from baselines.epf import extended_pf


def load_cinn_proposal(config_name, system):
    """
    使用原始 CINN_for_PF 中的 proposal_distribution 加载模型
    """
    try:
        from CINN_for_PF import proposal_distribution
        from Particle_filters.L96 import L96
        import os
        
        # 模型路径映射
        model_map = {
            'L96-50D-σv0.1-p1': '/root/CINN_PF/models_for_L96_test/model_1_1',
            'L96-50D-σv0.1-p2': '/root/CINN_PF/models_for_L96_test/model_2_1',
            'L96-50D-σv0.2-p1': '/root/CINN_PF/models_for_L96_test/model_1_2',
            'L96-50D-σv0.2-p2': '/root/CINN_PF/models_for_L96_test/model_2_2',
        }
        
        model_path = model_map.get(config_name)
        if model_path and os.path.exists(f'{model_path}/model.pth'):
            print(f"  [CINN] 加载预训练模型: {model_path}")
            
            # 创建与配置匹配的原始 L96 系统
            sigma_v = 0.1 if 'σv0.1' in config_name else 0.2
            p = 1 if 'p1' in config_name else 2
            orig_sys = L96(sigma_x=0.01, sigma_v=sigma_v, sigma_e=0.01, p=p, N=50)
            
            # 使用原始的 proposal_distribution
            q = proposal_distribution(orig_sys, load=model_path)
            return q, orig_sys
        
        print(f"  [CINN] 无可用模型: {config_name}")
    except Exception as e:
        print(f"  [CINN] 加载失败: {e}")
        import traceback
        traceback.print_exc()
    return None, None


def run_cinn_pf(data, orig_sys, proposal, num_particles=10):
    """
    使用原始 CINN_for_PF 实现运行 CINN-PF
    """
    if proposal is None or orig_sys is None:
        return None, None
    
    try:
        from CINN_for_PF import CINN_PF
        
        # 转换数据格式
        y = data['y']
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        u = data['u']
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
        
        # 构建原始格式的 data
        orig_data = {'u': u, 'y': y}
        
        # 使用原始 CINN_PF 函数
        x_est = CINN_PF(orig_data, orig_sys, proposal, num=num_particles)
        
        return x_est, None
    
    except Exception as e:
        print(f"  [CINN] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_method(method_name, data, system, x_true, **kwargs):
    """运行单个方法并返回 RMSE 和时间"""
    device = kwargs.get('device', 'cuda')
    
    t_start = time.time()
    
    try:
        if method_name == 'PF':
            result = standard_pf(data, system, num_particles=kwargs.get('n_particles', 500), device=device)
        elif method_name == 'EPF':
            result = extended_pf(data, system, num_particles=kwargs.get('n_particles', 100), device=device)
        elif method_name == 'SDPF':
            result = SDPF(data, system, 
                         num_particles=kwargs.get('n_particles', 20), 
                         n_steps=kwargs.get('n_steps', 10),
                         step_size=kwargs.get('step_size', 0.05),
                         device=device)
        elif method_name == 'CINN':
            proposal = kwargs.get('proposal')
            orig_sys = kwargs.get('orig_sys')
            if proposal is None or orig_sys is None:
                return float('nan'), float('nan')
            result, _ = run_cinn_pf(data, orig_sys, proposal, 
                                    num_particles=kwargs.get('n_particles', 10))
            if result is None:
                return float('nan'), float('nan')
        else:
            return float('nan'), float('nan')
        
        elapsed = time.time() - t_start
        
        # 转换为 NumPy
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        # RMSE
        x_est = result[1:]
        x_ref = x_true[1:]
        if x_est.shape[0] != x_ref.shape[0]:
            x_est = x_est[:x_ref.shape[0]]
        
        rmse = np.sqrt(np.mean((x_est - x_ref) ** 2))
        
        return rmse, elapsed
    
    except Exception as e:
        import traceback
        print(f"  [{method_name}] 错误: {e}")
        traceback.print_exc()
        return float('nan'), float('nan')


def run_scenario(name, system, n_steps, methods_config, cinn=None):
    """运行单个场景的所有方法对比"""
    print(f"\n{'='*70}")
    print(f"场景: {name}")
    print(f"维度: {system.dim}, 观测维度: {system.obs_dim}")
    print(f"{'='*70}")
    
    # 生成数据
    x_init = system.get_equilibrium() if hasattr(system, 'get_equilibrium') else np.random.randn(system.dim) * system.sigma_x
    data, x_true = system.generate(n_steps, x_init)
    
    results = {}
    
    for method, config in methods_config.items():
        print(f"\n运行 {method} (粒子数={config.get('n_particles', 'default')})...")
        
        rmse, elapsed = run_method(method, data, system, x_true, **config)
        
        if not np.isnan(rmse) and not np.isinf(rmse):
            print(f"  RMSE: {rmse:.4f}, 时间: {elapsed:.2f}s")
        else:
            print(f"  失败")
        
        results[method] = {'rmse': rmse, 'time': elapsed}
    
    return results


def main():
    print("=" * 70)
    print("高维场景全方法对比实验")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_results = {}
    
    # 方法配置
    methods_50d = {
        'PF': {'n_particles': 1000, 'device': device},
        'EPF': {'n_particles': 100, 'device': device},
        'SDPF': {'n_particles': 20, 'n_steps': 10, 'device': device},
        'CINN': {'n_particles': 10, 'device': device},
    }
    
    methods_highdim = {
        'PF': {'n_particles': 500, 'device': device},
        'EPF': {'n_particles': 50, 'device': device},
        'SDPF': {'n_particles': 20, 'n_steps': 10, 'device': device},
    }
    
    methods_extreme = {
        'PF': {'n_particles': 200, 'device': device},
        'SDPF': {'n_particles': 10, 'n_steps': 5, 'device': device},
    }
    
    # ===== 场景 1: L96-50D 基准 (有 CINN) =====
    print("\n\n" + "#" * 70)
    print("# 第一组: L96-50D 基准测试 (含 CINN-PF)")
    print("#" * 70)
    
    for sigma_v, p in [(0.1, 1), (0.1, 2), (0.2, 1), (0.2, 2)]:
        config_name = f'L96-50D-σv{sigma_v}-p{p}'
        sys_50d = L96System(dim=50, sigma_v=sigma_v, sigma_e=0.01, p=p)
        
        # 加载 CINN proposal
        proposal, orig_sys = load_cinn_proposal(config_name, sys_50d)
        
        # 更新方法配置以包含 proposal
        methods_50d_with_cinn = methods_50d.copy()
        methods_50d_with_cinn['CINN'] = {
            'n_particles': 10, 
            'proposal': proposal, 
            'orig_sys': orig_sys
        }
        
        res = run_scenario(config_name, sys_50d, n_steps=200, 
                          methods_config=methods_50d_with_cinn, cinn=None)
        all_results[config_name] = res
    
    # ===== 场景 2: 高维 L96 =====
    print("\n\n" + "#" * 70)
    print("# 第二组: 高维 L96 系统 (100D - 500D)")
    print("#" * 70)
    
    for dim in [100, 200, 500]:
        config_name = f'L96-{dim}D'
        sys_highdim = HighDimL96(dim=dim, sigma_v=0.1, sigma_e=0.01, p=1)
        
        n_steps = 100 if dim <= 200 else 50
        res = run_scenario(config_name, sys_highdim, n_steps=n_steps, methods_config=methods_highdim)
        all_results[config_name] = res
    
    # ===== 场景 3: 极限场景 =====
    print("\n\n" + "#" * 70)
    print("# 第三组: 极限场景测试")
    print("#" * 70)
    
    # 超高维
    sys_1000d = HighDimL96(dim=1000, sigma_v=0.1, sigma_e=0.01, p=1)
    res = run_scenario('L96-1000D', sys_1000d, n_steps=30, methods_config=methods_extreme)
    all_results['L96-1000D'] = res
    
    # 高噪声
    sys_noisy = HighDimL96(dim=50, sigma_v=0.5, sigma_e=0.1, p=1)
    res = run_scenario('高噪声(σv=0.5)', sys_noisy, n_steps=100, methods_config=methods_highdim)
    all_results['高噪声'] = res
    
    # 稀疏观测
    sys_sparse = HighDimL96(dim=100, sigma_v=0.1, sigma_e=0.01, p=1, obs_ratio=0.2)
    res = run_scenario('稀疏观测(20%)', sys_sparse, n_steps=100, methods_config=methods_highdim)
    all_results['稀疏观测'] = res
    
    # 强非线性
    sys_nonlin = HighDimL96(dim=50, sigma_v=0.1, sigma_e=0.01, p=3)
    res = run_scenario('强非线性(p=3)', sys_nonlin, n_steps=100, methods_config=methods_highdim)
    all_results['强非线性'] = res
    
    # ===== 汇总表格 =====
    print("\n\n" + "=" * 90)
    print("实验结果汇总")
    print("=" * 90)
    
    print(f"\n{'场景':<20} | {'PF':<12} | {'EPF':<12} | {'SDPF':<12} | {'CINN':<12} | {'最优':<8}")
    print("-" * 90)
    
    for scenario, res in all_results.items():
        pf = res.get('PF', {}).get('rmse', float('nan'))
        epf = res.get('EPF', {}).get('rmse', float('nan'))
        sdpf = res.get('SDPF', {}).get('rmse', float('nan'))
        cinn = res.get('CINN', {}).get('rmse', float('nan'))
        
        pf_str = f"{pf:.3f}" if not np.isnan(pf) else "N/A"
        epf_str = f"{epf:.3f}" if not np.isnan(epf) else "N/A"
        sdpf_str = f"{sdpf:.3f}" if not np.isnan(sdpf) else "N/A"
        cinn_str = f"{cinn:.3f}" if not np.isnan(cinn) else "N/A"
        
        # 找最优
        valid = [(v, n) for v, n in [(pf, 'PF'), (epf, 'EPF'), (sdpf, 'SDPF'), (cinn, 'CINN')] if not np.isnan(v)]
        best = min(valid, key=lambda x: x[0])[1] if valid else "N/A"
        
        print(f"{scenario:<20} | {pf_str:<12} | {epf_str:<12} | {sdpf_str:<12} | {cinn_str:<12} | {best:<8}")
    
    # 计算平均提升
    print("\n" + "-" * 90)
    print("方法对比分析:")
    
    sdpf_wins = 0
    total = 0
    sdpf_vs_pf = []
    sdpf_vs_epf = []
    sdpf_vs_cinn = []
    
    for scenario, res in all_results.items():
        pf = res.get('PF', {}).get('rmse', float('nan'))
        epf = res.get('EPF', {}).get('rmse', float('nan'))
        sdpf = res.get('SDPF', {}).get('rmse', float('nan'))
        cinn = res.get('CINN', {}).get('rmse', float('nan'))
        
        if not np.isnan(sdpf):
            total += 1
            if not np.isnan(pf):
                sdpf_vs_pf.append((pf - sdpf) / pf * 100)
            if not np.isnan(epf):
                sdpf_vs_epf.append((epf - sdpf) / epf * 100 if epf > 0 else 0)
            if not np.isnan(cinn):
                sdpf_vs_cinn.append((cinn - sdpf) / cinn * 100 if cinn > 0 else 0)
            
            valid = [(v, n) for v, n in [(pf, 'PF'), (epf, 'EPF'), (cinn, 'CINN')] if not np.isnan(v)]
            if valid and sdpf <= min(v for v, _ in valid):
                sdpf_wins += 1
    
    if sdpf_vs_pf:
        print(f"  SDPF vs PF:   平均提升 {np.mean(sdpf_vs_pf):.1f}%")
    if sdpf_vs_epf:
        print(f"  SDPF vs EPF:  平均提升 {np.mean(sdpf_vs_epf):.1f}%")
    if sdpf_vs_cinn:
        print(f"  SDPF vs CINN: 平均提升 {np.mean(sdpf_vs_cinn):.1f}%")
    
    print(f"\n  SDPF 最优场景数: {sdpf_wins}/{total}")
    
    print("\n注意:")
    print("- CINN-PF 仅在 L96-50D 上可用 (需要预训练模型)")
    print("- 高维场景 (>50D) 无 CINN 对比，因为没有预训练模型")
    print("- EPF 在高非线性 (p>1) 场景可能发散")


if __name__ == "__main__":
    main()
