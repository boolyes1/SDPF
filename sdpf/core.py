"""
SDPF 核心算法实现 (PyTorch 版本)

结构化扩散粒子滤波器 (Structured Diffusion Particle Filter)

完全基于 PyTorch 实现，支持 GPU 加速。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


def get_device(device: str = 'auto') -> torch.device:
    """获取计算设备"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class BaseParticleFilter(ABC):
    """粒子滤波器基类 (PyTorch)"""
    
    def __init__(self, system, num_particles: int = 100, device: str = 'auto'):
        """
        Args:
            system: 状态空间系统对象
            num_particles: 粒子数量
            device: 计算设备 ('auto', 'cuda', 'cpu')
        """
        self.system = system
        self.num_particles = num_particles
        self.device = get_device(device)
    
    @abstractmethod
    def run(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """运行粒子滤波"""
        pass


class StructuredDiffusionPF(BaseParticleFilter):
    """
    结构化扩散粒子滤波器 (PyTorch 完整版本)
    
    使用预条件化 Langevin 动力学生成高质量提议样本。
    完全基于 PyTorch，支持 GPU 加速。
    
    Args:
        system: 状态空间系统对象
        num_particles: 粒子数量 (默认 10)
        n_langevin_steps: Langevin 迭代步数 (默认 20)
        step_size: 初始步长 (默认 0.1)
        use_hessian: 是否使用 Hessian 预条件 (默认 True)
        annealing_rate: 步长退火率 (默认 0.95)
        device: 计算设备
    """
    
    def __init__(
        self, 
        system,
        num_particles: int = 10,
        n_langevin_steps: int = 20,
        step_size: float = 0.1,
        use_hessian: bool = True,
        annealing_rate: float = 0.95,
        device: str = 'auto'
    ):
        super().__init__(system, num_particles, device)
        
        self.n_steps = n_langevin_steps
        self.step_size = step_size
        self.use_hessian = use_hessian
        self.annealing_rate = annealing_rate
        
        # 系统参数
        self.dim = system.dim
        
        # 噪声协方差 (转换为 PyTorch tensor)
        self.Q = torch.tensor(system.Q, dtype=torch.float32, device=self.device)
        self.R = torch.tensor(system.R, dtype=torch.float32, device=self.device)
        self.Q_inv = torch.inverse(self.Q)
        self.R_inv = torch.inverse(self.R)
        
        # Cholesky 分解
        self.L_Q = torch.linalg.cholesky(self.Q)
    
    def f(self, x: torch.Tensor) -> torch.Tensor:
        """状态转移函数"""
        return self.system.f_torch(x)
    
    def h(self, x: torch.Tensor) -> torch.Tensor:
        """观测函数"""
        return self.system.h_torch(x)
    
    def compute_jacobian(
        self, 
        x: torch.Tensor, 
        func, 
        eps: float = 1e-5
    ) -> torch.Tensor:
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
    
    def compute_score(
        self, 
        x: torch.Tensor, 
        x_pred: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """计算 score 函数"""
        # 先验 score
        score_prior = -(x - x_pred) @ self.Q_inv
        
        # 似然 score
        h_x = self.h(x)
        diff_like = y - h_x
        
        J = self.compute_jacobian(x, self.h)
        
        score_like = torch.zeros_like(x)
        for i in range(x.shape[0]):
            score_like[i] = J[i].T @ self.R_inv @ diff_like[i]
        
        return score_prior + score_like
    
    def compute_preconditioner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算预条件矩阵的 Cholesky 分解"""
        num = x.shape[0]
        
        if not self.use_hessian:
            return self.L_Q.unsqueeze(0).expand(num, -1, -1)
        
        J = self.compute_jacobian(x, self.h)
        
        L_list = []
        for i in range(num):
            H = self.Q_inv + J[i].T @ self.R_inv @ J[i]
            try:
                M = torch.inverse(H)
                M = (M + M.T) / 2
                eigvals = torch.linalg.eigvalsh(M)
                if eigvals.min() < 1e-10:
                    M = M + (1e-10 - eigvals.min() + 1e-6) * torch.eye(self.dim, device=self.device)
                L = torch.linalg.cholesky(M)
            except:
                L = self.L_Q * 0.1
            L_list.append(L)
        
        return torch.stack(L_list)
    
    def langevin_sample(
        self, 
        x_init: torch.Tensor, 
        x_pred: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """Langevin 采样"""
        x = x_init.clone()
        num = x.shape[0]
        step = self.step_size
        
        for _ in range(self.n_steps):
            score = self.compute_score(x, x_pred, y)
            L = self.compute_preconditioner(x, y)
            
            # 预条件化梯度
            grad = torch.zeros_like(x)
            for i in range(num):
                grad[i] = L[i] @ (L[i].T @ score[i])
            
            # 结构化噪声
            noise = torch.randn(num, self.dim, device=self.device)
            structured_noise = torch.zeros_like(noise)
            for i in range(num):
                structured_noise[i] = L[i] @ noise[i]
            
            # 更新
            x = x + step * grad + (2 * step).sqrt() * structured_noise
            step *= self.annealing_rate
        
        return x
    
    def compute_weights(
        self, 
        x_new: torch.Tensor, 
        x_pred: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """计算归一化权重"""
        num = x_new.shape[0]
        
        # log p(y|x)
        h_x = self.h(x_new)
        diff_y = y - h_x
        log_like = torch.zeros(num, device=self.device)
        for i in range(num):
            log_like[i] = -0.5 * diff_y[i] @ self.R_inv @ diff_y[i]
        
        # 数值稳定归一化
        log_w = log_like - log_like.max()
        w = torch.exp(log_w)
        w = w / (w.sum() + 1e-300)
        
        if torch.isnan(w).any():
            w = torch.ones(num, device=self.device) / num
        
        return w
    
    def resample(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """重采样"""
        num = len(w)
        eff_n = 1.0 / (w ** 2).sum()
        
        if eff_n < num / 2:
            indices = torch.multinomial(w, num, replacement=True)
            return x[indices]
        return x
    
    def run(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """运行粒子滤波"""
        # 转换数据到设备
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
            y_t = y[t]
            
            # 预测
            x_pred = self.f(x)
            
            # Langevin 采样
            x_new = self.langevin_sample(x_pred, x_pred, y_t)
            
            # 权重
            w = self.compute_weights(x_new, x_pred, y_t)
            
            # 估计
            result[t] = (w.unsqueeze(1) * x_new).sum(dim=0)
            
            # 重采样
            x = self.resample(x_new, w)
        
        return result


def SDPF(
    data: Dict,
    system,
    num_particles: int = 10,
    n_steps: int = 20,
    step_size: float = 0.1,
    device: str = 'auto',
    **kwargs
) -> torch.Tensor:
    """SDPF 便捷函数"""
    pf = StructuredDiffusionPF(
        system,
        num_particles=num_particles,
        n_langevin_steps=n_steps,
        step_size=step_size,
        device=device,
        **kwargs
    )
    return pf.run(data)
