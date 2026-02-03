"""
Lorenz 96 混沌系统 (PyTorch 版本)

支持 NumPy 和 PyTorch 两种接口，可在 GPU 上运行。
"""

import torch
import numpy as np
from typing import Dict, Tuple

from .base import BaseSystem


class L96System(BaseSystem):
    """
    Lorenz 96 系统 (PyTorch 版本)
    
    状态转移 (RK4):
        x_t = RK4(x_{t-1}) + v_t
    
    观测模型:
        y_t = x_t + sin(p * x_{t,i-1} + p * x_{t,i+1}) + e_t
    
    Args:
        dim: 状态维度 (默认 50)
        sigma_v: 过程噪声标准差 (默认 0.1)
        sigma_e: 观测噪声标准差 (默认 0.01)
        sigma_x: 初始状态标准差 (默认 0.01)
        p: 观测非线性参数 (默认 1)
        F: 外部强迫参数 (默认 8.0)
        dt: 时间步长 (默认 0.01)
    """
    
    def __init__(
        self, 
        dim: int = 50,
        sigma_v: float = 0.1,
        sigma_e: float = 0.01,
        sigma_x: float = 0.01,
        p: int = 1,
        F: float = 8.0,
        dt: float = 0.01
    ):
        super().__init__(
            dim=dim,
            sigma_v=sigma_v,
            sigma_e=sigma_e,
            sigma_x=sigma_x
        )
        
        self.p = p
        self.F = F
        self.dt = dt
        
        # NumPy 索引
        self.idx_m2_np = np.arange(dim) - 2
        self.idx_m1_np = np.arange(dim) - 1
        self.idx_p1_np = (np.arange(dim) + 1) % dim
    
    def get_equilibrium(self) -> np.ndarray:
        """L96 平衡点"""
        return np.zeros(self.dim)
    
    # ============================================================
    # NumPy 实现
    # ============================================================
    
    def _l96_rhs_np(self, x: np.ndarray) -> np.ndarray:
        """L96 右端项 (NumPy)"""
        return (
            (x[:, self.idx_p1_np] - x[:, self.idx_m2_np]) * x[:, self.idx_m1_np] 
            - x 
            + self.F
        )
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """状态转移 RK4 (NumPy)"""
        k1 = self._l96_rhs_np(x)
        k2 = self._l96_rhs_np(x + k1 * self.dt / 2)
        k3 = self._l96_rhs_np(x + k2 * self.dt / 2)
        k4 = self._l96_rhs_np(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """观测函数 (NumPy)"""
        return x + np.sin(
            self.p * x[:, self.idx_m1_np] + self.p * x[:, self.idx_p1_np]
        )
    
    # ============================================================
    # PyTorch 实现
    # ============================================================
    
    def _get_torch_indices(self, device: torch.device):
        """获取 PyTorch 索引"""
        idx_m2 = torch.arange(self.dim, device=device) - 2
        idx_m1 = torch.arange(self.dim, device=device) - 1
        idx_p1 = (torch.arange(self.dim, device=device) + 1) % self.dim
        return idx_m2, idx_m1, idx_p1
    
    def _l96_rhs_torch(self, x: torch.Tensor) -> torch.Tensor:
        """L96 右端项 (PyTorch)"""
        idx_m2, idx_m1, idx_p1 = self._get_torch_indices(x.device)
        return (
            (x[:, idx_p1] - x[:, idx_m2]) * x[:, idx_m1] 
            - x 
            + self.F
        )
    
    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        """状态转移 RK4 (PyTorch)"""
        k1 = self._l96_rhs_torch(x)
        k2 = self._l96_rhs_torch(x + k1 * self.dt / 2)
        k3 = self._l96_rhs_torch(x + k2 * self.dt / 2)
        k4 = self._l96_rhs_torch(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        """观测函数 (PyTorch)"""
        idx_m2, idx_m1, idx_p1 = self._get_torch_indices(x.device)
        return x + torch.sin(self.p * x[:, idx_m1] + self.p * x[:, idx_p1])


def create_l96_system(
    dim: int = 50,
    sigma_v: float = 0.1,
    sigma_e: float = 0.01,
    p: int = 1
) -> L96System:
    """创建 L96 系统"""
    return L96System(dim=dim, sigma_v=sigma_v, sigma_e=sigma_e, p=p)
