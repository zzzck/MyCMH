"""
工具函数模块
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可复现

    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_name: str = 'auto') -> torch.device:
    """
    设置计算设备

    Args:
        device_name (str): 设备名称 ('auto', 'cpu', 'cuda', 'cuda:0', etc.)

    Returns:
        torch.device: 设备对象
    """
    if device_name == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_name)
        if device.type == 'cuda' and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            print(f"Using device: {device}")

    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量

    Args:
        model (nn.Module): 模型

    Returns:
        Dict[str, int]: 参数统计信息
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

    # 按模块统计
    module_params = {}
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = {
            'total': module_total,
            'trainable': module_trainable
        }

    param_info['module_parameters'] = module_params

    return param_info


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """
    打印模型信息

    Args:
        model (nn.Module): 模型
        model_name (str): 模型名称
    """
    param_info = count_parameters(model)

    print(f"\n{model_name} Information:")
    print("=" * 50)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable_parameters']:,}")

    print(f"\nModule-wise parameters:")
    for module_name, module_info in param_info['module_parameters'].items():
        print(f"  {module_name}: {module_info['trainable']:,} / {module_info['total']:,}")


def save_model(model: nn.Module,
               filepath: str,
               metadata: Optional[Dict[str, Any]] = None):
    """
    保存模型

    Args:
        model (nn.Module): 模型
        filepath (str): 保存路径
        metadata (Dict): 元数据
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }

    if metadata:
        save_dict['metadata'] = metadata

    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module,
               filepath: str,
               strict: bool = True) -> Dict[str, Any]:
    """
    加载模型

    Args:
        model (nn.Module): 模型
        filepath (str): 模型路径
        strict (bool): 是否严格匹配参数

    Returns:
        Dict[str, Any]: 加载的元数据
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    metadata = checkpoint.get('metadata', {})

    print(f"Model loaded from {filepath}")

    return metadata


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    计算模型大小

    Args:
        model (nn.Module): 模型

    Returns:
        Dict[str, float]: 模型大小信息（MB）
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': total_size / 1024 / 1024
    }


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds (float): 秒数

    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def create_directory(path: str):
    """
    创建目录

    Args:
        path (str): 目录路径
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str):
    """
    保存JSON文件

    Args:
        data (Dict): 数据
        filepath (str): 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    加载JSON文件

    Args:
        filepath (str): 文件路径

    Returns:
        Dict[str, Any]: 数据
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_gpu_memory_info():
    """
    获取GPU内存信息

    Returns:
        Dict[str, float]: GPU内存信息（GB）
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()

    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 3
    cached_memory = torch.cuda.memory_reserved(device) / 1024 ** 3
    free_memory = total_memory - cached_memory

    return {
        'total_gb': total_memory,
        'allocated_gb': allocated_memory,
        'cached_gb': cached_memory,
        'free_gb': free_memory
    }


def clear_gpu_memory():
    """
    清理GPU内存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AverageMeter:
    """
    平均值计算器
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新值

        Args:
            val: 新值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """
    计时器
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def stop(self):
        """停止计时"""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """
        获取经过的时间

        Returns:
            float: 经过的秒数
        """
        if self.start_time is None:
            return 0.0

        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def check_dependencies():
    """
    检查依赖包
    """
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'numpy',
        # 'pillow',
        # 'scikit-learn',
        'tqdm'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    else:
        print("All required packages are installed.")
        return True