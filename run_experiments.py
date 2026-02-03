#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDPF 完整实验脚本

运行: python run_experiments.py
"""

import numpy as np
from time import time

from systems import L96System
from sdpf.fast import FastSDPF
from baselines import StandardPF, ExtendedPF
from experiments import run_benchmark, compare_methods


def main():
    print()
    print("=" * 80)
    print("   SDPF 完整实验")
    print("   Lorenz 96 系统多配置对比")
    print("=" * 80)
    
    # 实验配置
    configs = [
        {'sigma_v': 0.1, 'p': 1, 'name': 'σv=0.1, p=1 (低噪声, 线性观测)'},
        {'sigma_v': 0.1, 'p': 2, 'name': 'σv=0.1, p=2 (低噪声, 非线性观测)'},
        {'sigma_v': 0.2, 'p': 1, 'name': 'σv=0.2, p=1 (高噪声, 线性观测)'},
        {'sigma_v': 0.2, 'p': 2, 'name': 'σv=0.2, p=2 (高噪声, 非线性观测)'},
    ]
    
    all_results = {}
    
    for cfg in configs:
        print(f"\n{'='*80}")
        print(f"配置: {cfg['name']}")
        print("=" * 80)
        
        # 创建系统
        system = L96System(
            dim=50, 
            sigma_v=cfg['sigma_v'], 
            sigma_e=0.01, 
            p=cfg['p']
        )
        
        # 定义方法
        def make_pf(d, s):
            return StandardPF(s, 1000).run(d)
        
        def make_sdpf(d, s):
            return FastSDPF(s, 10, n_steps=10, step_size=0.05).run(d)
        
        methods = {
            'PF-1000': make_pf,
            'SDPF-10': make_sdpf,
        }
        
        # 运行
        results = run_benchmark(system, methods, T=200, num_runs=5)
        summary = compare_methods(results, print_table=True)
        
        all_results[cfg['name']] = summary
    
    # 总结
    print("\n" + "=" * 80)
    print("总体结论")
    print("=" * 80)
    
    pf_rmses = []
    sdpf_rmses = []
    pf_times = []
    sdpf_times = []
    
    for name, summary in all_results.items():
        if 'PF-1000' in summary and not np.isnan(summary['PF-1000']['rmse_mean']):
            pf_rmses.append(summary['PF-1000']['rmse_mean'])
            pf_times.append(summary['PF-1000']['time_mean'])
        if 'SDPF-10' in summary and not np.isnan(summary['SDPF-10']['rmse_mean']):
            sdpf_rmses.append(summary['SDPF-10']['rmse_mean'])
            sdpf_times.append(summary['SDPF-10']['time_mean'])
    
    if pf_rmses and sdpf_rmses:
        avg_pf = np.mean(pf_rmses)
        avg_sdpf = np.mean(sdpf_rmses)
        improvement = (avg_pf - avg_sdpf) / avg_pf * 100
        speedup = np.mean(pf_times) / np.mean(sdpf_times)
        
        print(f"""
    ┌──────────────────────────────────────────────────────────────┐
    │                      SDPF 性能总结                           │
    ├──────────────────────────────────────────────────────────────┤
    │  PF-1000 平均 RMSE:    {avg_pf:.2f}                              │
    │  SDPF-10 平均 RMSE:    {avg_sdpf:.2f}                              │
    │                                                              │
    │  ★ RMSE 提升:          {improvement:.1f}%                           │
    │  ★ 粒子数减少:         100 倍 (1000 → 10)                    │
    │  ★ 速度提升:           {speedup:.1f} 倍                             │
    └──────────────────────────────────────────────────────────────┘
        """)
    
    print("\n✅ 实验完成!")


if __name__ == "__main__":
    main()
