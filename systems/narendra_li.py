"""
Narendra-Li 非线性系统

这是一个经典的2维非线性系统，用于测试粒子滤波器的性能。

参考: Narendra, K. S., & Li, S. M. (1996). Neural networks in control systems.

状态方程:
    x1' = x1/(1+x1²) + a1·sin(x2)
    x2' = x2·cos(x2) + x1·exp(-(x1²+x2²)/a2) + u³/(1+u²+a3·cos(x1+x2))

观测方程:
    y = x1/(1+b1·sin(x2)) + x2/(1+b2·sin(x1))
"""

import torch
import numpy as np
from typing import Dict, Tuple

from .base import BaseSystem


class NarendraLiSystem(BaseSystem):
    """
    Narendra-Li 非线性系统 (PyTorch 版本)
    
    Args:
        sigma_v: 过程噪声标准差 (默认 0.1)
        sigma_e: 观测噪声标准差 (默认 0.1)
        sigma_x: 初始状态标准差 (默认 0.1)
        par_f: 状态方程参数 [a1, a2, a3] (默认 [1.8, 8, 0.5])
        par_g: 观测方程参数 [b1, b2] (默认 [0.5, 0.5])
    """
    
    def __init__(
        self,
        sigma_v: float = 0.1,
        sigma_e: float = 0.1,
        sigma_x: float = 0.1,
        par_f: list = None,
        par_g: list = None
    ):
        super().__init__(
            dim=2,
            obs_dim=1,
            sigma_v=sigma_v,
            sigma_e=sigma_e,
            sigma_x=sigma_x
        )
        
        # 状态方程参数
        self.par_f = par_f if par_f is not None else [1.8, 8.0, 0.5]
        self.a1, self.a2, self.a3 = self.par_f
        
        # 观测方程参数
        self.par_g = par_g if par_g is not None else [0.5, 0.5]
        self.b1, self.b2 = self.par_g
    
    def get_equilibrium(self) -> np.ndarray:
        """初始平衡点"""
        return np.zeros(2)
    
    # ============================================================
    # NumPy 实现
    # ============================================================
    
    def f(self, x: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """
        状态转移函数 (NumPy)
        
        x1' = x1/(1+x1²) + a1·sin(x2)
        x2' = x2·cos(x2) + x1·exp(-(x1²+x2²)/a2) + u³/(1+u²+a3·cos(x1+x2))
        """
        if u is None:
            u = np.zeros((x.shape[0], 1))
        
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        x1_new = x1 / (1 + x1**2) + self.a1 * np.sin(x2)
        
        x2_new = (x2 * np.cos(x2) + 
                  x1 * np.exp(-(x1**2 + x2**2) / self.a2) +
                  u**3 / (1 + u**2 + self.a3 * np.cos(x1 + x2)))
        
        return np.hstack([x1_new, x2_new])
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        观测函数 (NumPy)
        
        y = x1/(1+b1·sin(x2)) + x2/(1+b2·sin(x1))
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        y = x1 / (1 + self.b1 * np.sin(x2)) + x2 / (1 + self.b2 * np.sin(x1))
        
        return y
    
    # ============================================================
    # PyTorch 实现
    # ============================================================
    
    def f_torch(self, x: torch.Tensor, u: torch.Tensor = None) -> torch.Tensor:
        """状态转移函数 (PyTorch)"""
        if u is None:
            u = torch.zeros((x.shape[0], 1), device=x.device)
        
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        x1_new = x1 / (1 + x1**2) + self.a1 * torch.sin(x2)
        
        x2_new = (x2 * torch.cos(x2) + 
                  x1 * torch.exp(-(x1**2 + x2**2) / self.a2) +
                  u**3 / (1 + u**2 + self.a3 * torch.cos(x1 + x2)))
        
        return torch.cat([x1_new, x2_new], dim=1)
    
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        """观测函数 (PyTorch)"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        y = x1 / (1 + self.b1 * torch.sin(x2)) + x2 / (1 + self.b2 * torch.sin(x1))
        
        return y
    
    # ============================================================
    # 数据生成 (重写以支持控制输入)
    # ============================================================
    
    def generate(
        self, 
        T: int, 
        u: np.ndarray = None,
        x0: np.ndarray = None,
        burn_in: int = 50
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        生成模拟数据
        
        Args:
            T: 时间步数
            u: 控制输入 (T, 1)，默认为随机输入
            x0: 初始状态
            burn_in: 燃烧期
        """
        # 控制输入
        if u is None:
            u = 2 * np.random.rand(T, 1) - 1  # U[-1, 1]
        
        # 初始化
        if x0 is None:
            x = self.get_equilibrium() + self.sigma_x * np.random.randn(self.dim)
        else:
            x = x0
        
        # 燃烧期
        for t in range(burn_in):
            u_t = 2 * np.random.rand(1, 1) - 1
            x = self.f(x[np.newaxis, :], u_t)[0] + self.sample_process_noise(1)[0]
        
        # 存储
        x_true = np.zeros((T + 1, self.dim))
        y = np.zeros((T + 1, self.obs_dim))
        
        x_true[0] = x
        y[0] = self.h(x[np.newaxis, :])[0] + self.sample_observation_noise(1)[0]
        
        for t in range(1, T + 1):
            u_t = u[t-1:t]
            x = self.f(x[np.newaxis, :], u_t)[0] + self.sample_process_noise(1)[0]
            x_true[t] = x
            y[t] = self.h(x[np.newaxis, :])[0] + self.sample_observation_noise(1)[0]
        
        return {'u': u, 'y': y}, x_true


def create_narendra_li_system(
    sigma_v: float = 0.1,
    sigma_e: float = 0.1
) -> NarendraLiSystem:
    """创建 Narendra-Li 系统"""
    return NarendraLiSystem(sigma_v=sigma_v, sigma_e=sigma_e)
