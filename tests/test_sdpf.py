"""
SDPF 单元测试
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from systems import L96System
from sdpf import SDPF
from sdpf.fast import FastSDPF


def test_l96_system():
    """测试 L96 系统"""
    system = L96System(dim=10, sigma_v=0.1)
    data, x_true = system.generate(T=10)
    
    assert x_true.shape == (11, 10)
    assert data['y'].shape == (11, 10)
    print("✓ L96 系统测试通过")


def test_sdpf_basic():
    """测试 SDPF 基本功能"""
    system = L96System(dim=10, sigma_v=0.1)
    data, x_true = system.generate(T=10)
    
    sdpf = FastSDPF(system, num_particles=5, n_steps=3)
    x_est = sdpf.run(data)
    
    assert x_est.shape == (11, 10)
    assert not np.isnan(x_est).any()
    print("✓ SDPF 基本功能测试通过")


def test_sdpf_accuracy():
    """测试 SDPF 精度"""
    np.random.seed(42)
    
    system = L96System(dim=20, sigma_v=0.1)
    data, x_true = system.generate(T=50)
    
    sdpf = FastSDPF(system, num_particles=10, n_steps=10)
    x_est = sdpf.run(data)
    
    rmse = np.sqrt(20 * np.mean((x_true[1:] - x_est[1:]) ** 2))
    
    # RMSE 应该合理 (< 20)
    assert rmse < 20, f"RMSE too high: {rmse}"
    print(f"✓ SDPF 精度测试通过 (RMSE={rmse:.2f})")


def test_reproducibility():
    """测试可重复性"""
    system = L96System(dim=10, sigma_v=0.1)
    data, _ = system.generate(T=10)
    
    np.random.seed(123)
    sdpf1 = FastSDPF(system, num_particles=5, n_steps=3)
    x1 = sdpf1.run(data)
    
    np.random.seed(123)
    sdpf2 = FastSDPF(system, num_particles=5, n_steps=3)
    x2 = sdpf2.run(data)
    
    assert np.allclose(x1, x2), "Results not reproducible"
    print("✓ 可重复性测试通过")


if __name__ == "__main__":
    print("\n运行 SDPF 单元测试...\n")
    
    test_l96_system()
    test_sdpf_basic()
    test_sdpf_accuracy()
    test_reproducibility()
    
    print("\n✅ 所有测试通过!")
