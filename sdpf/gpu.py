"""
SDPF GPU 版本 (PyTorch)

适用场景：
- 高步数 (T > 500)
- 大量粒子 (N > 100)
- 批量蒙特卡洛实验
- 需要 GPU 加速
"""

import torch
from typing import Dict, Union


def get_device(device: str = 'auto') -> torch.device:
    """获取计算设备"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class FastSDPF_GPU:
    """
    快速 SDPF (PyTorch/GPU 版本)
    
    完全向量化，支持 GPU 加速。
    
    Args:
        system: 状态空间系统
        num_particles: 粒子数 (默认 10)
        n_steps: Langevin 步数 (默认 10)
        step_size: 步长 (默认 0.05)
        annealing_rate: 退火率 (默认 0.9)
        device: 计算设备 ('auto', 'cuda', 'cpu')
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
        self.system = system
        self.num = num_particles
        self.n_steps = n_steps
        self.step_size = step_size
        self.annealing_rate = annealing_rate
        self.device = get_device(device)
        
        self.dim = system.dim
        self.obs_dim = getattr(system, 'obs_dim', system.dim)
        self.sigma_v = system.sigma_v
        self.sigma_e = system.sigma_e
        
        # 预计算 (在 GPU 上)
        self.prec_v = torch.tensor(1.0 / (self.sigma_v ** 2), device=self.device)
        self.prec_e = torch.tensor(1.0 / (self.sigma_e ** 2), device=self.device)
        self.precond = 1.0 / (self.prec_v + self.prec_e)
        self.L = torch.sqrt(self.precond)
        
        # 是否稀疏观测
        self.sparse_obs = (self.obs_dim != self.dim)
    
    @torch.no_grad()
    def run(self, data: Dict) -> torch.Tensor:
        """运行滤波"""
        # 数据转换
        if isinstance(data['u'], torch.Tensor):
            u = data['u'].to(self.device)
            y = data['y'].to(self.device)
        else:
            u = torch.tensor(data['u'], dtype=torch.float32, device=self.device)
            y = torch.tensor(data['y'], dtype=torch.float32, device=self.device)
        
        T = u.shape[0]
        
        # 初始化
        x = self.system.sample_initial_torch(self.num, self.device)
        result = torch.zeros((T + 1, self.dim), device=self.device)
        result[0] = x.mean(dim=0)
        
        for t in range(1, T + 1):
            y_t = y[t]
            
            # 预测
            x_pred = self.system.f_torch(x)
            
            # Langevin 采样
            x_new = x_pred.clone()
            step = self.step_size
            
            for _ in range(self.n_steps):
                # Score (向量化)
                h_x = self.system.h_torch(x_new)
                obs_error = y_t.unsqueeze(0) - h_x  # (num, obs_dim)
                
                if self.sparse_obs:
                    # 稀疏观测：需要将观测误差投影回状态空间
                    # 简化处理：只对被观测的维度施加观测修正
                    obs_score = torch.zeros(self.num, self.dim, device=self.device)
                    if hasattr(self.system, 'obs_indices'):
                        obs_idx = torch.tensor(self.system.obs_indices, device=self.device)
                        obs_score[:, obs_idx] = self.prec_e * obs_error
                    else:
                        # 假设观测前 obs_dim 个维度
                        obs_score[:, :self.obs_dim] = self.prec_e * obs_error
                else:
                    obs_score = self.prec_e * obs_error
                
                score = -self.prec_v * (x_new - x_pred) + obs_score
                
                # 更新
                noise = torch.randn(self.num, self.dim, device=self.device)
                x_new = x_new + step * self.precond * score + torch.sqrt(torch.tensor(2 * step, device=self.device)) * self.L * noise
                step *= self.annealing_rate
            
            # 权重
            h_x = self.system.h_torch(x_new)
            log_like = -0.5 * self.prec_e * ((y_t - h_x) ** 2).sum(dim=1)
            log_w = log_like - log_like.max()
            w = torch.exp(log_w)
            w = w / (w.sum() + 1e-30)
            
            if torch.isnan(w).any():
                w = torch.ones(self.num, device=self.device) / self.num
            
            # 估计
            result[t] = (w.unsqueeze(1) * x_new).sum(dim=0)
            
            # 重采样
            if 1.0 / (w ** 2).sum() < self.num / 2:
                indices = torch.multinomial(w, self.num, replacement=True)
                x = x_new[indices]
            else:
                x = x_new
        
        return result


def SDPF_GPU(
    data: Dict,
    system,
    num_particles: int = 10,
    n_steps: int = 10,
    step_size: float = 0.05,
    device: str = 'auto',
    **kwargs
) -> torch.Tensor:
    """
    SDPF GPU 版本便捷函数
    
    推荐用于:
    - 粒子数 > 100
    - 时间步 > 500
    - 批量实验
    - 需要 GPU 加速
    """
    pf = FastSDPF_GPU(
        system,
        num_particles=num_particles,
        n_steps=n_steps,
        step_size=step_size,
        device=device,
        **kwargs
    )
    return pf.run(data)
