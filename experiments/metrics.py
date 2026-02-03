"""
评估指标
"""

import numpy as np


def compute_rmse(x_true: np.ndarray, x_est: np.ndarray, dim: int = None) -> float:
    """
    计算 RMSE
    
    Args:
        x_true: 真实状态 (T, dim)
        x_est: 估计状态 (T, dim)
        dim: 状态维度 (用于归一化)
    
    Returns:
        RMSE 值
    """
    if dim is None:
        dim = x_true.shape[1] if x_true.ndim > 1 else 1
    
    mse = np.mean((x_true - x_est) ** 2)
    return np.sqrt(dim * mse)


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    计算有效样本量
    
    ESS = 1 / sum(w^2)
    
    Args:
        weights: 归一化权重
    
    Returns:
        ESS
    """
    return 1.0 / np.sum(weights ** 2)


def compute_nees(x_true: np.ndarray, x_est: np.ndarray, P: np.ndarray) -> float:
    """
    计算归一化估计误差平方 (Normalized Estimation Error Squared)
    
    NEES = (x_true - x_est)' * P^{-1} * (x_true - x_est)
    
    Args:
        x_true: 真实状态
        x_est: 估计状态
        P: 估计协方差
    
    Returns:
        NEES
    """
    diff = x_true - x_est
    P_inv = np.linalg.inv(P)
    return diff @ P_inv @ diff
