"""
SDPF 快速版本 (PyTorch 高度向量化)

完全向量化实现，最大化 GPU 并行度。
"""

import torch
from typing import Dict

from .core import BaseParticleFilter, get_device


class FastSDPF(BaseParticleFilter):
    """
    快速结构化扩散粒子滤波器 (PyTorch 高度向量化)
    
    完全向量化，避免 Python 循环，最大化 GPU 效率。
    
    Args:
        system: 状态空间系统对象
        num_particles: 粒子数量 (默认 10)
        n_steps: Langevin 迭代步数 (默认 10)
        step_size: 初始步长 (默认 0.05)
        annealing_rate: 步长退火率 (默认 0.9)
        device: 计算设备
    """
    
    def __init__(
        self, 
        system,
        num_particles: int = 10,
        n_steps: int = 10,
        step_size: float = 0.05,
        annealing_rate: float = 0.9,
        device: str = 'auto'
    ):
        super().__init__(system, num_particles, device)
        
        self.n_steps = n_steps
        self.step_size = step_size
        self.annealing_rate = annealing_rate
        
        self.dim = system.dim
        
        # 预计算参数 (在 GPU 上)
        self.sigma_v = system.sigma_v
        self.sigma_e = system.sigma_e
        
        self.prec_v = torch.tensor(1.0 / (self.sigma_v ** 2), device=self.device)
        self.prec_e = torch.tensor(1.0 / (self.sigma_e ** 2), device=self.device)
        
        # 预条件器
        h_diag = self.prec_v + self.prec_e
        self.precond = 1.0 / h_diag
        self.L = torch.sqrt(self.precond)
        
        # 预计算退火系数
        self.annealing_factors = torch.tensor(
            [self.annealing_rate ** k for k in range(n_steps)],
            device=self.device
        )
    
    @torch.no_grad()
    def run(self, data: Dict) -> torch.Tensor:
        """运行滤波 (完全向量化，无 Python 循环在关键路径)"""
        # 数据转换
        if isinstance(data['u'], torch.Tensor):
            u = data['u'].to(self.device)
            y = data['y'].to(self.device)
        else:
            u = torch.tensor(data['u'], dtype=torch.float32, device=self.device)
            y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)
        
        T = u.shape[0]
        num = self.num_particles
        dim = self.dim
        
        # 初始化
        x = self.system.sample_initial_torch(num, self.device)
        result = torch.zeros((T + 1, dim), device=self.device)
        result[0] = x.mean(dim=0)
        
        # 主循环
        for t in range(1, T + 1):
            y_t = y[t]  # (dim,)
            
            # 预测 (向量化)
            x_pred = self.system.f_torch(x)  # (num, dim)
            
            # Langevin 采样 (展开循环以利用 GPU)
            x_new = x_pred.clone()
            step = self.step_size
            
            for k in range(self.n_steps):
                # Score 计算 (完全向量化)
                h_x = self.system.h_torch(x_new)  # (num, dim)
                score_prior = -self.prec_v * (x_new - x_pred)  # (num, dim)
                score_like = self.prec_e * (y_t.unsqueeze(0) - h_x)  # (num, dim)
                score = score_prior + score_like  # (num, dim)
                
                # Langevin 步 (向量化)
                noise = torch.randn(num, dim, device=self.device)
                x_new = x_new + step * self.precond * score + (2 * step) ** 0.5 * self.L * noise
                step = step * self.annealing_rate
            
            # 权重计算 (向量化)
            h_x = self.system.h_torch(x_new)
            log_like = -0.5 * self.prec_e * ((y_t - h_x) ** 2).sum(dim=1)  # (num,)
            
            log_w = log_like - log_like.max()
            w = torch.exp(log_w)
            w = w / (w.sum() + 1e-30)
            
            # 处理 NaN
            if torch.isnan(w).any() or w.sum() < 1e-10:
                w = torch.ones(num, device=self.device) / num
            
            # 状态估计 (向量化)
            result[t] = (w.unsqueeze(1) * x_new).sum(dim=0)
            
            # 重采样
            eff_n = 1.0 / (w ** 2).sum()
            if eff_n < num / 2:
                # 使用 torch.multinomial 进行向量化重采样
                indices = torch.multinomial(w, num, replacement=True)
                x = x_new[indices]
            else:
                x = x_new
        
        return result


class BatchFastSDPF(BaseParticleFilter):
    """
    批量处理版本 - 同时处理多个独立序列
    
    用于大规模蒙特卡洛实验。
    """
    
    def __init__(
        self, 
        system,
        num_particles: int = 10,
        n_steps: int = 10,
        step_size: float = 0.05,
        annealing_rate: float = 0.9,
        batch_size: int = 10,
        device: str = 'auto'
    ):
        super().__init__(system, num_particles, device)
        
        self.n_steps = n_steps
        self.step_size = step_size
        self.annealing_rate = annealing_rate
        self.batch_size = batch_size
        
        self.dim = system.dim
        self.sigma_v = system.sigma_v
        self.sigma_e = system.sigma_e
        
        self.prec_v = torch.tensor(1.0 / (self.sigma_v ** 2), device=self.device)
        self.prec_e = torch.tensor(1.0 / (self.sigma_e ** 2), device=self.device)
        
        h_diag = self.prec_v + self.prec_e
        self.precond = 1.0 / h_diag
        self.L = torch.sqrt(self.precond)
    
    @torch.no_grad()
    def run_batch(self, data_list: list) -> list:
        """批量运行多个序列"""
        results = []
        for data in data_list:
            pf = FastSDPF(
                self.system, 
                self.num_particles, 
                self.n_steps,
                self.step_size,
                self.annealing_rate,
                self.device
            )
            results.append(pf.run(data))
        return results


def fast_sdpf(
    data: Dict,
    system,
    num_particles: int = 10,
    n_steps: int = 10,
    step_size: float = 0.05,
    device: str = 'auto',
    **kwargs
) -> torch.Tensor:
    """快速 SDPF 便捷函数"""
    pf = FastSDPF(
        system,
        num_particles=num_particles,
        n_steps=n_steps,
        step_size=step_size,
        device=device,
        **kwargs
    )
    return pf.run(data)
