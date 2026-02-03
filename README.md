# SDPF - 结构化扩散粒子滤波器

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SDPF** (Structured Diffusion Particle Filter) 是一种**无需预训练**的高效粒子滤波方法。

参考论文: Sun et al., "A conditional invertible neural network-based particle filter", *Automatica* 183 (2026)

---

## ✨ 核心优势

| 特性 | SDPF | CINN-PF | 标准 PF |
|------|------|---------|---------|
| 预训练 | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| 粒子数 | 10 | 10 | 10,000+ |
| RMSE | ~3-5 | ~4-5 | ~15-25 |
| 在线自适应 | ✅ | ❌ | ✅ |

---

## 📊 性能对比 (Lorenz 96, 50维)

基于论文 [Sun et al., Automatica 2026] 的实验设置：

| 方法 | 粒子数 | σv=0.1,p=1 | σv=0.1,p=2 | σv=0.2,p=1 | σv=0.2,p=2 | 预训练 |
|------|--------|------------|------------|------------|------------|--------|
| PF | 10,000 | ~10-15 | ~15-20 | ~10-15 | ~15-25 | 不需要 |
| EPF | 100 | ~5-8 | 发散 | ~8-12 | 发散 | 不需要 |
| UPF | 100 | ~5-8 | 发散 | ~8-12 | 发散 | 不需要 |
| A-IPF | 10 | ~5-7 | ~5-7 | ~6-8 | ~6-8 | 不需要 |
| CINN-PF | 10 | ~4-5 | ~4-5 | ~4-6 | ~4-6 | **需要** |
| **SDPF** | 10 | **~3-4** | **~3-4** | **~4-5** | **~4-5** | **不需要** |

**关键发现**: SDPF 用 10 个粒子达到 CINN-PF 的精度，且无需预训练！

---

## 🚀 快速开始

### 安装

```bash
git clone <repo>
cd SDPF
pip install -r requirements.txt
```

### 运行演示

```bash
python run_demo.py              # CPU/GPU 双版本演示
python run_paper_comparison.py  # 论文完整对比
```

### 代码示例

```python
from systems import L96System
from sdpf import SDPF, SDPF_CPU, SDPF_GPU

# 创建系统
system = L96System(dim=50, sigma_v=0.1, sigma_e=0.01, p=1)

# 生成数据
data, x_true = system.generate(T=200)

# 方法1: 自动选择 CPU/GPU
x_est = SDPF(data, system, num_particles=10, device='auto')

# 方法2: 显式选择 CPU (小规模推荐)
x_est = SDPF_CPU(data, system, num_particles=10)

# 方法3: 显式选择 GPU (大规模推荐)
x_est = SDPF_GPU(data, system, num_particles=100, device='cuda')

# 计算 RMSE
import numpy as np
rmse = np.sqrt(50 * np.mean((x_true[1:] - x_est[1:]) ** 2))
print(f"RMSE: {rmse:.2f}")
```

---

## 📁 项目结构

```
SDPF/
├── sdpf/                    # 核心算法
│   ├── __init__.py          # 统一接口
│   ├── cpu.py               # CPU 版本 (NumPy)
│   ├── gpu.py               # GPU 版本 (PyTorch)
│   ├── core.py              # 完整实现
│   └── fast.py              # 快速实现
│
├── systems/                 # 状态空间系统
│   ├── base.py              # 基类 (NumPy + PyTorch)
│   └── lorenz96.py          # L96 系统
│
├── baselines/               # 基线方法
│   ├── pf.py                # 标准 PF
│   └── epf.py               # 扩展 PF
│
├── experiments/             # 实验工具
│   ├── benchmark.py
│   └── metrics.py
│
├── run_demo.py              # 快速演示
├── run_paper_comparison.py  # 论文对比实验
├── requirements.txt
└── README.md
```

---

## 🔬 算法原理

SDPF 使用**预条件化 Langevin 动力学**从后验分布采样：

```
对每个时间步 t:
  1. 预测: x_pred = f(x_{t-1})
  
  2. Langevin 采样 (K 步):
     score = -Q⁻¹(x - x_pred) + H'R⁻¹(y - h(x))   # Score 函数
     M = (Q⁻¹ + H'R⁻¹H)⁻¹                         # 预条件矩阵
     x ← x + ε·M·score + √(2ε·M)·noise            # Langevin 步
     
  3. 计算权重: w ∝ p(y|x)
  4. 重采样
```

**与 CINN-PF 的区别**:
- CINN-PF: 离线训练神经网络 → 在线用神经网络采样
- SDPF: 完全在线计算 → 使用 Langevin 动力学采样

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_particles` | 10 | 粒子数量 |
| `n_steps` | 10 | Langevin 迭代步数 |
| `step_size` | 0.05 | 初始步长 |
| `annealing_rate` | 0.9 | 步长退火率 |
| `device` | 'auto' | 计算设备 |

### 版本选择建议

| 场景 | 推荐版本 | 原因 |
|------|----------|------|
| 粒子数 < 100, T < 500 | `SDPF_CPU` | 无 GPU 开销 |
| 粒子数 > 100, T > 500 | `SDPF_GPU` | 并行优势 |
| 批量蒙特卡洛实验 | `SDPF_GPU` | 吞吐量高 |

---

## 📚 参考文献

1. **CINN-PF**: Sun, W., Xiong, W., Huang, B., & Chen, H. (2026). A conditional invertible neural network-based particle filter. *Automatica*, 183, 112639.

2. **Langevin Monte Carlo**: Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions.

3. **Particle Filtering**: Doucet, A., de Freitas, N., & Gordon, N. (2001). Sequential Monte Carlo methods in practice.

4. **Diffusion Model Sampling Mechanism: Transient Chaotic Dynamics Perspective**:Bu,Y.,Lian,D.&Gao,Z.(2026).Under Review
---

## 📝 许可证

MIT License
