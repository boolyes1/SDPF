"""
SDPF - 结构化扩散粒子滤波器
Structured Diffusion Particle Filter

提供两个版本：
- CPU 版本 (NumPy): 适用于低步数、小规模、快速原型
- GPU 版本 (PyTorch): 适用于高步数、大规模、批量实验

使用示例:
    # CPU 版本 (推荐用于: 粒子数<100, 步数<500)
    >>> from sdpf import SDPF_CPU
    >>> x_est = SDPF_CPU(data, system, num_particles=10)
    
    # GPU 版本 (推荐用于: 粒子数>100, 步数>500, 批量实验)
    >>> from sdpf import SDPF_GPU
    >>> x_est = SDPF_GPU(data, system, num_particles=100, device='cuda')
    
    # 自动选择版本
    >>> from sdpf import SDPF
    >>> x_est = SDPF(data, system, num_particles=10, device='auto')
"""

from .cpu import SDPF_CPU, FastSDPF_CPU
from .gpu import SDPF_GPU, FastSDPF_GPU

import torch


def SDPF(data, system, num_particles=10, n_steps=10, step_size=0.05, device='auto', **kwargs):
    """
    SDPF 统一接口 - 自动选择 CPU/GPU 版本
    
    Args:
        data: 数据字典 {'u': 控制, 'y': 观测}
        system: 状态空间系统
        num_particles: 粒子数
        n_steps: Langevin 步数
        step_size: 步长
        device: 'auto', 'cpu', 'cuda'
    
    Returns:
        状态估计 (numpy array 或 torch tensor)
    
    选择策略:
        - device='cpu' 或无 CUDA: 使用 CPU 版本
        - device='cuda' 或 'auto'+有CUDA: 使用 GPU 版本
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        return SDPF_CPU(data, system, num_particles, n_steps, step_size, **kwargs)
    else:
        return SDPF_GPU(data, system, num_particles, n_steps, step_size, device, **kwargs)


__all__ = [
    'SDPF',
    'SDPF_CPU', 'FastSDPF_CPU',
    'SDPF_GPU', 'FastSDPF_GPU',
]

__version__ = '1.0.0'
