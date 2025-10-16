#!/usr/bin/env python3
"""
跨模态哈希检索模型训练脚本

使用示例:
    python train.py --config configs/synthetic_config.json
    python train.py --dataset synthetic --hash_dim 64 --batch_size 32
"""

import argparse
import os
# 设置环境变量
os.environ['HTTP_PROXY'] = 'http://211.81.248.212:3128'
os.environ['HTTPS_PROXY'] = 'http://211.81.248.212:3128'
os.environ['http_proxy'] = 'http://211.81.248.212:3128'
os.environ['https_proxy'] = 'http://211.81.248.212:3128'

import sys
import torch
from torch.utils.data import DataLoader, random_split

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cross_modal_hash import CrossModalHashModel
from models.text_encoder import TextTokenizer
from data.dataset import SyntheticDataset, COCODataset, Flickr30KDataset
from data.dataloader import create_dataloader
from data.transforms import get_transforms
from training.config import TrainingConfig, get_synthetic_config, get_coco_config, get_flickr30k_config
from training.trainer import CrossModalHashTrainer
from utils.utils import set_seed, setup_device, print_model_info, check_dependencies
from utils.logger import setup_logger


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Cross-Modal Hash Retrieval Training')

    # 基本配置
    parser.add_argument('--config', type=str, help='配置文件路径', default='configs/synthetic_config.json')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'coco', 'flickr30k'], help='数据集类型')
    parser.add_argument('--data_dir', type=str, help='数据集目录')
    parser.add_argument('--annotations_file', type=str, help='标注文件路径')

    # 模型配置
    parser.add_argument('--hash_dim', type=int, default=64, help='哈希码维度')
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--text_model', type=str, default='bert-base-uncased', help='文本模型')
    parser.add_argument('--image_backbone', type=str, default='resnet50', help='图像骨干网络')
    parser.add_argument('--hash_activation', type=str, default='tanh', help='哈希激活函数')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')

    # 损失配置
    parser.add_argument('--lambda_quant', type=float, default=0.1, help='量化损失权重')
    parser.add_argument('--lambda_balance', type=float, default=0.01, help='平衡损失权重')

    # 设备配置
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--mixed_precision', action='store_true', help='是否使用混合精度')

    # 其他配置
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--resume_from', type=str, help='从检查点恢复训练')

    return parser.parse_args()


def create_dataset(config, tokenizer, split='train'):
    """
    创建数据集

    Args:
        config (TrainingConfig): 训练配置
        tokenizer: 文本分词器
        split (str): 数据集分割

    Returns:
        Dataset: 数据集
    """
    # 获取图像变换
    is_training = (split == 'train')
    image_transform = get_transforms(
        image_size=config.image_size,
        is_training=is_training,
        augment_strength=config.augment_strength
    )

    if config.dataset_name == 'synthetic':
        # 合成数据集
        num_samples = 1000 if split == 'train' else 200
        dataset = SyntheticDataset(
            num_samples=num_samples,
            num_classes=10,
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_text_length=config.max_text_length
        )

    elif config.dataset_name == 'coco':
        # COCO数据集
        dataset = COCODataset(
            image_dir=config.image_dir,
            annotations_file=config.annotations_file,
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_text_length=config.max_text_length,
            split=split
        )

    elif config.dataset_name == 'flickr30k':
        # Flickr30K数据集
        dataset = Flickr30KDataset(
            image_dir=config.image_dir,
            annotations_file=config.annotations_file,
            tokenizer=tokenizer,
            image_transform=image_transform,
            max_text_length=config.max_text_length,
            split=split
        )

    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    return dataset


def create_dataloaders(config):
    """
    创建数据加载器

    Args:
        config (TrainingConfig): 训练配置

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # 创建分词器
    tokenizer = TextTokenizer(
        model_name=config.text_model,
        max_length=config.max_text_length
    )

    if config.dataset_name == 'synthetic':
        # 合成数据集：创建训练、验证、测试集
        full_dataset = create_dataset(config, tokenizer, 'train')

        # 分割数据集
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    else:
        # 真实数据集：分别创建
        train_dataset = create_dataset(config, tokenizer, 'train')
        val_dataset = create_dataset(config, tokenizer, 'val')
        test_dataset = create_dataset(config, tokenizer, 'test')

    # 创建数据加载器
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )

    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    return train_dataloader, val_dataloader, test_dataloader


def main():
    """
    主训练函数
    """
    # 解析参数
    args = parse_args()

    # 检查依赖
    if not check_dependencies():
        return

    # 加载配置
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        # 根据数据集类型获取默认配置
        if args.dataset == 'synthetic':
            config = get_synthetic_config()
        elif args.dataset == 'coco':
            config = get_coco_config()
        elif args.dataset == 'flickr30k':
            config = get_flickr30k_config()
        else:
            config = TrainingConfig()

    # 更新配置（命令行参数优先）
    config.update(
        dataset_name=args.dataset,
        hash_dim=args.hash_dim,
        feature_dim=args.feature_dim,
        text_model=args.text_model,
        image_backbone=args.image_backbone,
        hash_activation=args.hash_activation,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_quant=args.lambda_quant,
        lambda_balance=args.lambda_balance,
        device=args.device,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from
    )

    # 更新数据路径
    if args.data_dir:
        config.image_dir = args.data_dir
    if args.annotations_file:
        config.annotations_file = args.annotations_file

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = setup_device(config.device)
    config.device = str(device)

    # 设置日志
    logger = setup_logger(
        'training',
        os.path.join(config.log_dir, 'training.log')
    )

    logger.info("Starting Cross-Modal Hash Retrieval Training")
    logger.info(f"Configuration: {config.to_dict()}")

    # 创建模型
    model = CrossModalHashModel(
        hash_dim=config.hash_dim,
        feature_dim=config.feature_dim,
        text_model=config.text_model,
        image_backbone=config.image_backbone,
        hash_activation=config.hash_activation,
        use_adaptive_hash=config.use_adaptive_hash,
        temperature=config.temperature
    )

    # 打印模型信息
    print_model_info(model, "Cross-Modal Hash Model")

    # 创建数据加载器
    logger.info("Creating datasets and dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)

    logger.info(f"Train samples: {len(train_dataloader.dataset)}")
    logger.info(f"Val samples: {len(val_dataloader.dataset)}")
    logger.info(f"Test samples: {len(test_dataloader.dataset)}")

    # 创建训练器
    trainer = CrossModalHashTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(filename='interrupted_checkpoint.pth')
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
