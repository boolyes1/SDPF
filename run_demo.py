#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDPF 演示 - CPU/GPU 双版本对比
"""

import numpy as np
import torch
from time import time

from systems import L96System
from sdpf import SDPF, SDPF_CPU, SDPF_GPU
from baselines import StandardPF


def main():
    print()
    print("=" * 70)
    print("   SDPF - CPU/GPU 双版本演示")
    print("=" * 70)
    
    # 检测设备
    has_cuda = torch.cuda.is_available()
    print(f"\n🖥️ 可用设备:")
    print(f"   • CPU: ✓")
    print(f"   • GPU: {'✓ ' + torch.cuda.get_device_name(0) if has_cuda else '✗'}")
    
    # 创建系统
    system = L96System(dim=50, sigma_v=0.1, sigma_e=0.01, p=1)
    
    print(f"\n📊 系统配置: L96, dim=50, σv=0.1, σe=0.01")
    
    # 生成数据
    print("\n🔄 生成数据 (T=200)...")
    T = 200
    data, x_true = system.generate(T)
    
    print("\n" + "=" * 70)
    print("🧪 测试 1: CPU 版本 (适用于低步数/小规模)")
    print("=" * 70)
    
    # CPU PF
    print("\n[CPU] PF-1000...")
    pf_cpu = StandardPF(system, num_particles=1000, device='cpu')
    start = time()
    x_pf_cpu = pf_cpu.run(data)
    t_pf_cpu = time() - start
    if isinstance(x_pf_cpu, torch.Tensor):
        x_pf_cpu = x_pf_cpu.cpu().numpy()
    rmse_pf_cpu = np.sqrt(50 * np.mean((x_true[1:] - x_pf_cpu[1:]) ** 2))
    print(f"   RMSE={rmse_pf_cpu:.2f}, Time={t_pf_cpu:.3f}s")
    
    # CPU SDPF
    print("\n[CPU] SDPF-10...")
    start = time()
    x_sdpf_cpu = SDPF_CPU(data, system, num_particles=10, n_steps=10)
    t_sdpf_cpu = time() - start
    rmse_sdpf_cpu = np.sqrt(50 * np.mean((x_true[1:] - x_sdpf_cpu[1:]) ** 2))
    print(f"   RMSE={rmse_sdpf_cpu:.2f}, Time={t_sdpf_cpu:.3f}s")
    
    if has_cuda:
        print("\n" + "=" * 70)
        print("🧪 测试 2: GPU 版本 (适用于高步数/大规模)")
        print("=" * 70)
        
        # GPU 预热
        _ = torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')
        torch.cuda.synchronize()
        
        # GPU PF
        print("\n[GPU] PF-1000...")
        pf_gpu = StandardPF(system, num_particles=1000, device='cuda')
        torch.cuda.synchronize()
        start = time()
        x_pf_gpu = pf_gpu.run(data)
        torch.cuda.synchronize()
        t_pf_gpu = time() - start
        x_pf_gpu_np = x_pf_gpu.cpu().numpy()
        rmse_pf_gpu = np.sqrt(50 * np.mean((x_true[1:] - x_pf_gpu_np[1:]) ** 2))
        print(f"   RMSE={rmse_pf_gpu:.2f}, Time={t_pf_gpu:.3f}s")
        
        # GPU SDPF
        print("\n[GPU] SDPF-10...")
        torch.cuda.synchronize()
        start = time()
        x_sdpf_gpu = SDPF_GPU(data, system, num_particles=10, n_steps=10, device='cuda')
        torch.cuda.synchronize()
        t_sdpf_gpu = time() - start
        x_sdpf_gpu_np = x_sdpf_gpu.cpu().numpy()
        rmse_sdpf_gpu = np.sqrt(50 * np.mean((x_true[1:] - x_sdpf_gpu_np[1:]) ** 2))
        print(f"   RMSE={rmse_sdpf_gpu:.2f}, Time={t_sdpf_gpu:.3f}s")
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("📈 结果汇总")
    print("=" * 70)
    
    print(f"\n{'方法':<15} | {'设备':<6} | {'RMSE':<8} | {'时间':<10}")
    print("-" * 50)
    print(f"{'PF-1000':<15} | {'CPU':<6} | {rmse_pf_cpu:<8.2f} | {t_pf_cpu:<10.3f}s")
    print(f"{'SDPF-10 ★':<15} | {'CPU':<6} | {rmse_sdpf_cpu:<8.2f} | {t_sdpf_cpu:<10.3f}s")
    
    if has_cuda:
        print(f"{'PF-1000':<15} | {'GPU':<6} | {rmse_pf_gpu:<8.2f} | {t_pf_gpu:<10.3f}s")
        print(f"{'SDPF-10 ★':<15} | {'GPU':<6} | {rmse_sdpf_gpu:<8.2f} | {t_sdpf_gpu:<10.3f}s")
    
    # 使用建议
    print("\n" + "=" * 70)
    print("💡 版本选择建议")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  场景                      │  推荐版本     │  原因              │
    ├─────────────────────────────────────────────────────────────────┤
    │  快速测试/原型             │  CPU          │  启动快，无开销    │
    │  粒子数 < 100, T < 500     │  CPU          │  计算量小          │
    │  粒子数 > 100, T > 500     │  GPU          │  并行优势明显      │
    │  批量蒙特卡洛实验          │  GPU          │  吞吐量高          │
    │  无 GPU 环境               │  CPU          │  唯一选择          │
    └─────────────────────────────────────────────────────────────────┘
    
    使用方式:
        from sdpf import SDPF_CPU, SDPF_GPU, SDPF
        
        # 显式选择
        x = SDPF_CPU(data, system, num_particles=10)   # CPU 版本
        x = SDPF_GPU(data, system, num_particles=10)   # GPU 版本
        
        # 自动选择 (推荐)
        x = SDPF(data, system, num_particles=10, device='auto')
    """)
    
    print("=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
