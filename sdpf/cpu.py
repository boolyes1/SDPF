"""
SDPF CPU 版本 (NumPy)

适用场景：
- 低步数 (T < 500)
- 少量粒子 (N < 100)
- 快速原型和调试
- 无 GPU 环境
"""

import numpy as np
from typing import Dict


class FastSDPF_CPU:
    """
    快速 SDPF (NumPy/CPU 版本)
    
    使用向量化 NumPy 计算，适用于小规模问题。
    
    Args:
        system: 状态空间系统
        num_particles: 粒子数 (默认 10)
        n_steps: Langevin 步数 (默认 10)
        step_size: 步长 (默认 0.05)
        annealing_rate: 退火率 (默认 0.9)
    """
    
    def __init__(
        self, 
        system,
        num_particles: int = 10,
        n_steps: int = 10,
        step_size: float = 0.05,
        annealing_rate: float = 0.9
    ):
        self.system = system
        self.num = num_particles
        self.n_steps = n_steps
        self.step_size = step_size
        self.annealing_rate = annealing_rate
        
        self.dim = system.dim
        self.sigma_v = system.sigma_v
        self.sigma_e = system.sigma_e
        
        # 预计算
        self.prec_v = 1.0 / (self.sigma_v ** 2)
        self.prec_e = 1.0 / (self.sigma_e ** 2)
        self.precond = 1.0 / (self.prec_v + self.prec_e)
        self.L = np.sqrt(self.precond)
    
    def run(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """运行滤波"""
        u = data['u']
        y = data['y']
        T = u.shape[0]
        
        # 初始化
        x = self.system.sample_initial(self.num)
        result = np.zeros((T + 1, self.dim))
        result[0] = np.mean(x, axis=0)
        
        for t in range(1, T + 1):
            y_t = y[t]
            
            # 预测
            x_pred = self.system.f(x)
            
            # Langevin 采样
            x_new = x_pred.copy()
            step = self.step_size
            
            for _ in range(self.n_steps):
                # Score (向量化)
                h_x = self.system.h(x_new)
                score = -self.prec_v * (x_new - x_pred) + self.prec_e * (y_t - h_x)
                
                # 更新
                noise = np.random.randn(self.num, self.dim)
                x_new = x_new + step * self.precond * score + np.sqrt(2 * step) * self.L * noise
                step *= self.annealing_rate
            
            # 权重
            h_x = self.system.h(x_new)
            log_like = -0.5 * self.prec_e * np.sum((y_t - h_x) ** 2, axis=1)
            log_w = log_like - np.max(log_like)
            w = np.exp(log_w)
            w = np.clip(w, 1e-300, None)
            w = w / np.sum(w)
            
            if np.any(np.isnan(w)):
                w = np.ones(self.num) / self.num
            
            # 估计
            result[t] = np.sum(w[:, np.newaxis] * x_new, axis=0)
            
            # 重采样
            if 1.0 / np.sum(w ** 2) < self.num / 2:
                indices = np.random.choice(self.num, size=self.num, p=w)
                x = x_new[indices]
            else:
                x = x_new
        
        return result


def SDPF_CPU(
    data: Dict[str, np.ndarray],
    system,
    num_particles: int = 10,
    n_steps: int = 10,
    step_size: float = 0.05,
    **kwargs
) -> np.ndarray:
    """
    SDPF CPU 版本便捷函数
    
    推荐用于:
    - 粒子数 < 100
    - 时间步 < 500
    - 快速测试和原型
    """
    pf = FastSDPF_CPU(
        system,
        num_particles=num_particles,
        n_steps=n_steps,
        step_size=step_size,
        **kwargs
    )
    return pf.run(data)
