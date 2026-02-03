"""
高维状态估计测试系统

包含多个常用的高维基准系统：
1. 高维 L96 (100维、200维)
2. 耦合 Lorenz 63 系统
3. 稀疏观测系统
4. 高维线性系统
"""

import torch
import numpy as np
from typing import Dict, Tuple, List

try:
    from .base import BaseSystem
except ImportError:
    from base import BaseSystem


class HighDimL96(BaseSystem):
    """
    高维 Lorenz 96 系统
    
    支持任意维度，用于测试粒子滤波器的维度扩展性。
    
    Args:
        dim: 状态维度 (默认 100)
        sigma_v: 过程噪声 (默认 0.1)
        sigma_e: 观测噪声 (默认 0.01)
        p: 观测非线性参数 (默认 1)
        F: 外部强迫 (默认 8.0)
        obs_ratio: 观测比例，1.0表示全观测 (默认 1.0)
    """
    
    def __init__(
        self,
        dim: int = 100,
        sigma_v: float = 0.1,
        sigma_e: float = 0.01,
        p: int = 1,
        F: float = 8.0,
        dt: float = 0.01,
        obs_ratio: float = 1.0
    ):
        # 观测维度
        obs_dim = max(1, int(dim * obs_ratio))
        
        super().__init__(
            dim=dim,
            obs_dim=obs_dim,
            sigma_v=sigma_v,
            sigma_e=sigma_e,
            sigma_x=0.01
        )
        
        self.p = p
        self.F = F
        self.dt = dt
        self.obs_ratio = obs_ratio
        
        # 观测索引 (均匀分布)
        self.obs_indices = np.linspace(0, dim-1, obs_dim, dtype=int)
        
        # NumPy 索引
        self.idx_m2 = np.arange(dim) - 2
        self.idx_m1 = np.arange(dim) - 1
        self.idx_p1 = (np.arange(dim) + 1) % dim
    
    def _l96_rhs(self, x: np.ndarray) -> np.ndarray:
        """L96 右端项"""
        return (
            (x[:, self.idx_p1] - x[:, self.idx_m2]) * x[:, self.idx_m1] 
            - x + self.F
        )
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """RK4 积分"""
        k1 = self._l96_rhs(x)
        k2 = self._l96_rhs(x + k1 * self.dt / 2)
        k3 = self._l96_rhs(x + k2 * self.dt / 2)
        k4 = self._l96_rhs(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """观测函数 (可能是稀疏的)"""
        x_obs = x[:, self.obs_indices]
        if self.p > 0:
            # 非线性观测
            idx_m1 = (self.obs_indices - 1) % self.dim
            idx_p1 = (self.obs_indices + 1) % self.dim
            return x_obs + np.sin(self.p * x[:, idx_m1] + self.p * x[:, idx_p1])
        return x_obs
    
    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 状态转移"""
        device = x.device
        idx_m2 = torch.arange(self.dim, device=device) - 2
        idx_m1 = torch.arange(self.dim, device=device) - 1
        idx_p1 = (torch.arange(self.dim, device=device) + 1) % self.dim
        
        def rhs(x):
            return (x[:, idx_p1] - x[:, idx_m2]) * x[:, idx_m1] - x + self.F
        
        k1 = rhs(x)
        k2 = rhs(x + k1 * self.dt / 2)
        k3 = rhs(x + k2 * self.dt / 2)
        k4 = rhs(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 观测函数"""
        device = x.device
        obs_idx = torch.tensor(self.obs_indices, device=device)
        x_obs = x[:, obs_idx]
        
        if self.p > 0:
            idx_m1 = (obs_idx - 1) % self.dim
            idx_p1 = (obs_idx + 1) % self.dim
            return x_obs + torch.sin(self.p * x[:, idx_m1] + self.p * x[:, idx_p1])
        return x_obs


class CoupledLorenz63(BaseSystem):
    """
    耦合 Lorenz 63 系统
    
    多个 Lorenz 63 系统通过耦合项连接，形成高维混沌系统。
    
    Lorenz 63 方程:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    Args:
        n_systems: 耦合的子系统数量 (默认 10)
        coupling: 耦合强度 (默认 0.1)
        sigma_v: 过程噪声 (默认 0.1)
        sigma_e: 观测噪声 (默认 1.0)
    """
    
    def __init__(
        self,
        n_systems: int = 10,
        coupling: float = 0.1,
        sigma_v: float = 0.1,
        sigma_e: float = 1.0,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        dt: float = 0.01
    ):
        dim = 3 * n_systems
        
        super().__init__(
            dim=dim,
            obs_dim=dim,
            sigma_v=sigma_v,
            sigma_e=sigma_e,
            sigma_x=1.0
        )
        
        self.n_systems = n_systems
        self.coupling = coupling
        self.sigma_l = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
    
    def get_equilibrium(self) -> np.ndarray:
        """初始点在吸引子附近"""
        eq = np.zeros(self.dim)
        for i in range(self.n_systems):
            eq[3*i] = 1.0 + 0.1 * np.random.randn()
            eq[3*i+1] = 1.0 + 0.1 * np.random.randn()
            eq[3*i+2] = 1.0 + 0.1 * np.random.randn()
        return eq
    
    def _lorenz63_rhs(self, state: np.ndarray) -> np.ndarray:
        """计算右端项"""
        num = state.shape[0]
        rhs = np.zeros_like(state)
        
        for i in range(self.n_systems):
            x = state[:, 3*i]
            y = state[:, 3*i+1]
            z = state[:, 3*i+2]
            
            # Lorenz 63 方程
            rhs[:, 3*i] = self.sigma_l * (y - x)
            rhs[:, 3*i+1] = x * (self.rho - z) - y
            rhs[:, 3*i+2] = x * y - self.beta * z
            
            # 耦合项 (与相邻系统)
            if i > 0:
                x_prev = state[:, 3*(i-1)]
                rhs[:, 3*i] += self.coupling * (x_prev - x)
            if i < self.n_systems - 1:
                x_next = state[:, 3*(i+1)]
                rhs[:, 3*i] += self.coupling * (x_next - x)
        
        return rhs
    
    def f(self, x: np.ndarray) -> np.ndarray:
        """RK4 积分"""
        k1 = self._lorenz63_rhs(x)
        k2 = self._lorenz63_rhs(x + k1 * self.dt / 2)
        k3 = self._lorenz63_rhs(x + k2 * self.dt / 2)
        k4 = self._lorenz63_rhs(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """观测：直接观测所有状态"""
        return x
    
    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 版本"""
        device = x.device
        num = x.shape[0]
        rhs = torch.zeros_like(x)
        
        for i in range(self.n_systems):
            xi = x[:, 3*i]
            yi = x[:, 3*i+1]
            zi = x[:, 3*i+2]
            
            rhs[:, 3*i] = self.sigma_l * (yi - xi)
            rhs[:, 3*i+1] = xi * (self.rho - zi) - yi
            rhs[:, 3*i+2] = xi * yi - self.beta * zi
            
            if i > 0:
                rhs[:, 3*i] += self.coupling * (x[:, 3*(i-1)] - xi)
            if i < self.n_systems - 1:
                rhs[:, 3*i] += self.coupling * (x[:, 3*(i+1)] - xi)
        
        k1 = rhs
        k2 = self._lorenz63_rhs_torch(x + k1 * self.dt / 2)
        k3 = self._lorenz63_rhs_torch(x + k2 * self.dt / 2)
        k4 = self._lorenz63_rhs_torch(x + k3 * self.dt)
        return x + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _lorenz63_rhs_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch RHS"""
        rhs = torch.zeros_like(x)
        for i in range(self.n_systems):
            xi = x[:, 3*i]
            yi = x[:, 3*i+1]
            zi = x[:, 3*i+2]
            rhs[:, 3*i] = self.sigma_l * (yi - xi)
            rhs[:, 3*i+1] = xi * (self.rho - zi) - yi
            rhs[:, 3*i+2] = xi * yi - self.beta * zi
            if i > 0:
                rhs[:, 3*i] += self.coupling * (x[:, 3*(i-1)] - xi)
            if i < self.n_systems - 1:
                rhs[:, 3*i] += self.coupling * (x[:, 3*(i+1)] - xi)
        return rhs
    
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HighDimLinear(BaseSystem):
    """
    高维线性系统
    
    用于基准测试，了解方法在线性系统上的表现。
    
    x_{t+1} = A @ x_t + v_t
    y_t = H @ x_t + e_t
    
    Args:
        dim: 状态维度 (默认 100)
        obs_dim: 观测维度 (默认 50)
        stability: 系统稳定性参数，<1 稳定 (默认 0.95)
    """
    
    def __init__(
        self,
        dim: int = 100,
        obs_dim: int = 50,
        stability: float = 0.95,
        sigma_v: float = 0.1,
        sigma_e: float = 0.1
    ):
        super().__init__(
            dim=dim,
            obs_dim=obs_dim,
            sigma_v=sigma_v,
            sigma_e=sigma_e,
            sigma_x=0.1
        )
        
        # 生成稳定的状态转移矩阵
        # 使用带状矩阵结构
        self.A = np.zeros((dim, dim))
        for i in range(dim):
            self.A[i, i] = stability
            if i > 0:
                self.A[i, i-1] = (1 - stability) / 2
            if i < dim - 1:
                self.A[i, i+1] = (1 - stability) / 2
        
        # 观测矩阵 (随机选择观测的状态)
        self.obs_indices = np.sort(np.random.choice(dim, obs_dim, replace=False))
        self.H = np.zeros((obs_dim, dim))
        for i, idx in enumerate(self.obs_indices):
            self.H[i, idx] = 1.0
        
        # PyTorch 版本
        self.A_torch = None
        self.H_torch = None
    
    def f(self, x: np.ndarray) -> np.ndarray:
        return x @ self.A.T
    
    def h(self, x: np.ndarray) -> np.ndarray:
        return x @ self.H.T
    
    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.A_torch is None or self.A_torch.device != x.device:
            self.A_torch = torch.tensor(self.A, dtype=torch.float32, device=x.device)
        return x @ self.A_torch.T
    
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.H_torch is None or self.H_torch.device != x.device:
            self.H_torch = torch.tensor(self.H, dtype=torch.float32, device=x.device)
        return x @ self.H_torch.T
