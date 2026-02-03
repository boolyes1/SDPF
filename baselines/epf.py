"""
扩展粒子滤波器 (PyTorch 版本)
"""

import torch
from typing import Dict


class ExtendedPF:
    """
    扩展粒子滤波器 (PyTorch)
    
    Args:
        system: 状态空间系统对象
        num_particles: 粒子数量 (默认 100)
        device: 计算设备
    """
    
    def __init__(self, system, num_particles: int = 100, device: str = 'auto'):
        self.system = system
        self.num_particles = num_particles
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.dim = system.dim
        
        self.Q = torch.tensor(system.Q, dtype=torch.float32, device=self.device)
        self.R = torch.tensor(system.R, dtype=torch.float32, device=self.device)
        self.Q_inv = torch.inverse(self.Q)
        self.R_inv = torch.inverse(self.R)
    
    def compute_jacobian(self, x: torch.Tensor, func, eps: float = 1e-5) -> torch.Tensor:
        """数值计算 Jacobian"""
        num = x.shape[0]
        out_dim = func(x).shape[1]
        in_dim = x.shape[1]
        
        J = torch.zeros((num, out_dim, in_dim), device=self.device)
        for j in range(in_dim):
            x_plus = x.clone()
            x_plus[:, j] += eps
            x_minus = x.clone()
            x_minus[:, j] -= eps
            J[:, :, j] = (func(x_plus) - func(x_minus)) / (2 * eps)
        
        return J
    
    def run(self, data: Dict) -> torch.Tensor:
        """运行滤波"""
        if isinstance(data['u'], torch.Tensor):
            u = data['u'].to(self.device)
            y = data['y'].to(self.device)
        else:
            u = torch.tensor(data['u'], dtype=torch.float32, device=self.device)
            y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)
        
        T = u.shape[0]
        
        x = self.system.sample_initial_torch(self.num_particles, self.device)
        result = torch.zeros((T + 1, self.dim), device=self.device)
        result[0] = x.mean(dim=0)
        
        for t in range(1, T + 1):
            x_pred = self.system.f_torch(x)
            H = self.compute_jacobian(x_pred, self.system.h_torch)
            
            x_new = torch.zeros_like(x)
            log_w = torch.zeros(self.num_particles, device=self.device)
            
            for i in range(self.num_particles):
                # 后验精度
                P_inv = self.Q_inv + H[i].T @ self.R_inv @ H[i]
                try:
                    P = torch.inverse(P_inv)
                except:
                    P = self.Q
                
                # 后验均值
                h_pred = self.system.h_torch(x_pred[[i]])[0]
                innovation = y[t] - h_pred
                K = P @ H[i].T @ self.R_inv
                m = x_pred[i] + K @ innovation
                
                # 采样
                try:
                    L = torch.linalg.cholesky(P + 1e-6 * torch.eye(self.dim, device=self.device))
                    x_new[i] = m + L @ torch.randn(self.dim, device=self.device)
                except:
                    x_new[i] = m + 0.01 * torch.randn(self.dim, device=self.device)
                
                # 权重
                log_w[i] = -0.5 * innovation @ self.R_inv @ innovation
            
            # 归一化
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = torch.clamp(w, min=1e-300)
            w = w / w.sum()
            
            if torch.isnan(w).any():
                w = torch.ones(self.num_particles, device=self.device) / self.num_particles
            
            result[t] = (w.unsqueeze(1) * x_new).sum(dim=0)
            
            # 重采样
            eff_n = 1.0 / (w ** 2).sum()
            if eff_n < self.num_particles / 2:
                indices = torch.multinomial(w, self.num_particles, replacement=True)
                x = x_new[indices]
            else:
                x = x_new
        
        return result


def extended_pf(data, system, num_particles: int = 100, device: str = 'auto') -> torch.Tensor:
    """EPF 便捷函数"""
    pf = ExtendedPF(system, num_particles, device)
    return pf.run(data)
