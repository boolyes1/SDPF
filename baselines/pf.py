"""
标准粒子滤波器 (PyTorch 版本)
"""

import torch
from typing import Dict


class StandardPF:
    """
    标准粒子滤波器 (PyTorch)
    
    Args:
        system: 状态空间系统对象
        num_particles: 粒子数量 (默认 1000)
        device: 计算设备
    """
    
    def __init__(self, system, num_particles: int = 1000, device: str = 'auto'):
        self.system = system
        self.num_particles = num_particles
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.dim = system.dim
        self.sigma_v = system.sigma_v
        self.sigma_e = system.sigma_e
        self.prec_e = 1.0 / (self.sigma_e ** 2)
    
    def compute_weights(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算归一化权重"""
        h_x = self.system.h_torch(x)
        log_like = -0.5 * self.prec_e * ((y - h_x) ** 2).sum(dim=1)
        
        log_w = log_like - log_like.max()
        w = torch.exp(log_w)
        w = torch.clamp(w, min=1e-300)
        w = w / w.sum()
        
        if torch.isnan(w).any():
            w = torch.ones(self.num_particles, device=self.device) / self.num_particles
        
        return w
    
    def run(self, data: Dict) -> torch.Tensor:
        """运行滤波"""
        if isinstance(data['u'], torch.Tensor):
            u = data['u'].to(self.device)
            y = data['y'].to(self.device)
        else:
            u = torch.tensor(data['u'], dtype=torch.float32, device=self.device)
            y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)
        
        T = u.shape[0]
        
        # 初始化
        x = self.system.sample_initial_torch(self.num_particles, self.device)
        result = torch.zeros((T + 1, self.dim), device=self.device)
        result[0] = x.mean(dim=0)
        
        for t in range(1, T + 1):
            # 采样 (从先验)
            noise = self.system.sample_process_noise_torch(self.num_particles, self.device)
            x_new = self.system.f_torch(x) + noise
            
            # 权重
            w = self.compute_weights(x_new, y[t])
            
            # 估计
            result[t] = (w.unsqueeze(1) * x_new).sum(dim=0)
            
            # 重采样
            eff_n = 1.0 / (w ** 2).sum()
            if eff_n < self.num_particles / 2:
                indices = torch.multinomial(w, self.num_particles, replacement=True)
                x = x_new[indices]
            else:
                x = x_new
        
        return result


def standard_pf(data, system, num_particles: int = 1000, device: str = 'auto') -> torch.Tensor:
    """标准 PF 便捷函数"""
    pf = StandardPF(system, num_particles, device)
    return pf.run(data)
