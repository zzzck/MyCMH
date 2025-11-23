"""
训练配置模块
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os


@dataclass
class TrainingConfig:
    """
    训练配置类
    """

    # 模型配置
    hash_dim: int = 64
    feature_dim: int = 512
    text_model: str = 'bert-base-uncased'
    image_backbone: str = 'resnet50'
    hash_activation: str = 'tanh'
    use_adaptive_hash: bool = False
    temperature: float = 1.0

    # 数据配置
    dataset_name: str = 'synthetic'  # 'coco', 'flickr30k', 'synthetic'
    image_dir: str = ''
    annotations_file: str = ''
    image_size: int = 224
    max_text_length: int = 128
    augment_strength: str = 'medium'

    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'linear'

    # 损失配置
    margin: float = 0.2
    lambda_quant: float = 0.1
    lambda_balance: float = 0.01
    contrastive_temperature: float = 0.07

    # 优化器配置
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # 数据加载配置
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    use_balanced_sampler: bool = False
    samples_per_class: int = 2

    # 评估配置
    eval_interval: int = 1
    eval_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])

    # 保存和日志配置
    save_dir: str = '/workspace/cross_modal_hash_retrieval/checkpoints'
    log_dir: str = '/workspace/cross_modal_hash_retrieval/logs'
    save_interval: int = 1
    log_interval: int = 100

    # 设备配置
    device: str = 'cuda'
    mixed_precision: bool = True
    distributed: bool = False
    local_rank: int = 0

    # 其他配置
    seed: int = 42
    resume_from: Optional[str] = None
    pretrained_weights: Optional[str] = None

    def save(self, filepath: str):
        """
        保存配置到文件

        Args:
            filepath (str): 文件路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 转换为字典
        config_dict = self.__dict__.copy()

        # 保存为JSON
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """
        从文件加载配置

        Args:
            filepath (str): 文件路径

        Returns:
            TrainingConfig: 配置对象
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def update(self, **kwargs):
        """
        更新配置

        Args:
            **kwargs: 要更新的配置项
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config key: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.__dict__.copy()


def get_default_config():
    """
    获取默认配置

    Returns:
        TrainingConfig: 默认配置
    """
    return TrainingConfig()


def get_coco_config():
    """
    获取COCO数据集配置

    Returns:
        TrainingConfig: COCO配置
    """
    config = TrainingConfig()
    config.dataset_name = 'coco'
    config.batch_size = 64
    config.num_epochs = 200
    config.learning_rate = 2e-4
    config.hash_dim = 128
    config.feature_dim = 1024
    config.image_backbone = 'resnet101'
    return config


def get_flickr30k_config():
    """
    获取Flickr30K数据集配置

    Returns:
        TrainingConfig: Flickr30K配置
    """
    config = TrainingConfig()
    config.dataset_name = 'flickr30k'
    config.batch_size = 48
    config.num_epochs = 150
    config.learning_rate = 1.5e-4
    config.hash_dim = 96
    config.feature_dim = 768
    return config


def get_synthetic_config():
    """
    获取合成数据集配置（用于测试）

    Returns:
        TrainingConfig: 合成数据集配置
    """
    config = TrainingConfig()
    config.dataset_name = 'synthetic'
    config.batch_size = 16
    config.num_epochs = 50
    config.learning_rate = 1e-3
    config.hash_dim = 32
    config.feature_dim = 256
    config.eval_interval = 2
    config.save_interval = 5
    return config