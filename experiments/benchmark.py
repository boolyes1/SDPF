"""
基准测试和方法对比
"""

import numpy as np
from time import time
from typing import Dict, List, Callable, Any

from .metrics import compute_rmse


def run_benchmark(
    system,
    methods: Dict[str, Callable],
    T: int = 200,
    num_runs: int = 5,
    verbose: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    运行基准测试
    
    Args:
        system: 状态空间系统
        methods: 方法字典 {'name': func}
        T: 时间步数
        num_runs: 运行次数
        verbose: 是否打印进度
    
    Returns:
        results: {'method': {'rmse': [...], 'time': [...]}}
    """
    results = {name: {'rmse': [], 'time': []} for name in methods}
    
    for run in range(num_runs):
        if verbose:
            print(f"Run {run + 1}/{num_runs}")
        
        # 生成数据
        data, x_true = system.generate(T)
        
        for name, method in methods.items():
            try:
                start = time()
                x_est = method(data, system)
                elapsed = time() - start
                
                rmse = compute_rmse(x_true[1:], x_est[1:], system.dim)
                
                results[name]['rmse'].append(rmse)
                results[name]['time'].append(elapsed)
                
                if verbose:
                    print(f"  {name}: RMSE={rmse:.2f}, Time={elapsed:.2f}s")
            except Exception as e:
                if verbose:
                    print(f"  {name}: Failed - {e}")
                results[name]['rmse'].append(np.nan)
                results[name]['time'].append(np.nan)
    
    return results


def compare_methods(
    results: Dict[str, Dict[str, List[float]]],
    print_table: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    汇总对比结果
    
    Args:
        results: run_benchmark 的输出
        print_table: 是否打印表格
    
    Returns:
        summary: {'method': {'rmse_mean': x, 'rmse_std': x, 'time_mean': x}}
    """
    summary = {}
    
    for name, data in results.items():
        rmses = [r for r in data['rmse'] if not np.isnan(r)]
        times = [t for t in data['time'] if not np.isnan(t)]
        
        summary[name] = {
            'rmse_mean': np.mean(rmses) if rmses else np.nan,
            'rmse_std': np.std(rmses) if rmses else np.nan,
            'time_mean': np.mean(times) if times else np.nan,
            'success_rate': len(rmses) / len(data['rmse'])
        }
    
    if print_table:
        print("\n" + "=" * 70)
        print(f"{'Method':<15} | {'RMSE':<15} | {'Time':<10} | {'Success':<8}")
        print("-" * 70)
        
        for name, s in summary.items():
            rmse_str = f"{s['rmse_mean']:.2f} ± {s['rmse_std']:.2f}" if not np.isnan(s['rmse_mean']) else "N/A"
            time_str = f"{s['time_mean']:.3f}s" if not np.isnan(s['time_mean']) else "N/A"
            success_str = f"{s['success_rate']*100:.0f}%"
            print(f"{name:<15} | {rmse_str:<15} | {time_str:<10} | {success_str:<8}")
        
        print("=" * 70)
    
    return summary
