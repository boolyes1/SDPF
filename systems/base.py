"""
状态空间系统基类 (PyTorch 版本)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union


class BaseSystem(ABC):
    """
    状态空间系统基类
    
    系统形式:
        x_t = f(x_{t-1}) + v_t,  v_t ~ N(0, Q)
        y_t = h(x_t) + e_t,      e_t ~ N(0, R)
    
    支持 NumPy 和 PyTorch 两种接口。
    """
    
    def __init__(
        self, 
        dim: int,
        obs_dim: int = None,
        sigma_v: float = 0.1,
        sigma_e: float = 0.01,
        sigma_x: float = 0.01
    ):
        self.dim = dim
        self.obs_dim = obs_dim if obs_dim is not None else dim
        
        self.sigma_v = sigma_v
        self.sigma_e = sigma_e
        self.sigma_x = sigma_x
        
        # NumPy 协方差矩阵
        self.Q = (sigma_v ** 2) * np.eye(dim)
        self.R = (sigma_e ** 2) * np.eye(self.obs_dim)
    
    # ============================================================
    # NumPy 接口
    # ============================================================
    
    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """状态转移 (NumPy)"""
        pass
    
    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """观测函数 (NumPy)"""
        pass
    
    def get_equilibrium(self) -> np.ndarray:
        """平衡点"""
        return np.zeros(self.dim)
    
    def sample_initial(self, num: int) -> np.ndarray:
        """采样初始状态 (NumPy)"""
        return self.get_equilibrium() + self.sigma_x * np.random.randn(num, self.dim)
    
    def sample_process_noise(self, num: int) -> np.ndarray:
        """采样过程噪声 (NumPy)"""
        return self.sigma_v * np.random.randn(num, self.dim)
    
    def sample_observation_noise(self, num: int) -> np.ndarray:
        """采样观测噪声 (NumPy)"""
        return self.sigma_e * np.random.randn(num, self.obs_dim)
    
    # ============================================================
    # PyTorch 接口
    # ============================================================
    
    @abstractmethod
    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        """状态转移 (PyTorch)"""
        pass
    
    @abstractmethod
    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        """观测函数 (PyTorch)"""
        pass
    
    def sample_initial_torch(self, num: int, device: torch.device) -> torch.Tensor:
        """采样初始状态 (PyTorch)"""
        eq = torch.tensor(self.get_equilibrium(), dtype=torch.float32, device=device)
        return eq + self.sigma_x * torch.randn(num, self.dim, device=device)
    
    def sample_process_noise_torch(self, num: int, device: torch.device) -> torch.Tensor:
        """采样过程噪声 (PyTorch)"""
        return self.sigma_v * torch.randn(num, self.dim, device=device)
    
    def sample_observation_noise_torch(self, num: int, device: torch.device) -> torch.Tensor:
        """采样观测噪声 (PyTorch)"""
        return self.sigma_e * torch.randn(num, self.obs_dim, device=device)
    
    # ============================================================
    # 数据生成
    # ============================================================
    
    def generate(
        self, 
        T: int, 
        x0: np.ndarray = None,
        burn_in: int = 100
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """生成模拟数据 (NumPy)"""
        if x0 is None:
            x = self.get_equilibrium() + self.sigma_x * np.random.randn(self.dim)
        else:
            x = x0
        
        # 燃烧期
        for _ in range(burn_in):
            x = self.f(x[np.newaxis, :])[0] + self.sample_process_noise(1)[0]
        
        # 存储
        x_true = np.zeros((T + 1, self.dim))
        y = np.zeros((T + 1, self.obs_dim))
        u = np.zeros((T, 1))
        
        x_true[0] = x
        y[0] = self.h(x[np.newaxis, :])[0] + self.sample_observation_noise(1)[0]
        
        for t in range(1, T + 1):
            x = self.f(x[np.newaxis, :])[0] + self.sample_process_noise(1)[0]
            x_true[t] = x
            y[t] = self.h(x[np.newaxis, :])[0] + self.sample_observation_noise(1)[0]
        
        return {'u': u, 'y': y}, x_true
    
    def generate_torch(
        self, 
        T: int, 
        device: torch.device,
        burn_in: int = 100
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """生成模拟数据 (PyTorch)"""
        data, x_true = self.generate(T, burn_in=burn_in)
        
        return {
            'u': torch.tensor(data['u'], dtype=torch.float32, device=device),
            'y': torch.tensor(data['y'], dtype=torch.float32, device=device)
        }, torch.tensor(x_true, dtype=torch.float32, device=device)
