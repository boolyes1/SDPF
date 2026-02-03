"""
默认配置
"""

# 系统配置
SYSTEM_CONFIG = {
    'dim': 50,
    'sigma_v': 0.1,
    'sigma_e': 0.01,
    'sigma_x': 0.01,
    'p': 1,
    'F': 8.0,
    'dt': 0.05,
}

# SDPF 配置
SDPF_CONFIG = {
    'num_particles': 10,
    'n_steps': 10,
    'step_size': 0.05,
    'annealing_rate': 0.9,
}

# PF 配置
PF_CONFIG = {
    'num_particles': 1000,
}

# 实验配置
EXPERIMENT_CONFIG = {
    'T': 200,
    'num_runs': 10,
}
