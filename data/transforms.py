"""
数据变换模块
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def get_transforms(image_size=224, is_training=True, augment_strength='medium'):
    """
    获取图像变换

    Args:
        image_size (int): 图像尺寸
        is_training (bool): 是否为训练模式
        augment_strength (str): 数据增强强度 ('weak', 'medium', 'strong')

    Returns:
        torchvision.transforms.Compose: 图像变换
    """
    if is_training:
        if augment_strength == 'weak':
            transform = transforms.Compose([
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif augment_strength == 'medium':
            transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif augment_strength == 'strong':
            transform = transforms.Compose([
                transforms.Resize((image_size + 64, image_size + 64)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported augment_strength: {augment_strength}")
    else:
        # 测试时的变换
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


class TwoCropsTransform:
    """
    双重裁剪变换，用于对比学习
    """

    def __init__(self, base_transform):
        """
        初始化双重裁剪变换

        Args:
            base_transform: 基础变换
        """
        self.base_transform = base_transform

    def __call__(self, x):
        """
        应用双重裁剪

        Args:
            x: 输入图像

        Returns:
            list: 两个变换后的图像
        """
        return [self.base_transform(x), self.base_transform(x)]


class GaussianBlur:
    """
    高斯模糊变换
    """

    def __init__(self, sigma=[0.1, 2.0]):
        """
        初始化高斯模糊

        Args:
            sigma (list): 模糊程度范围
        """
        self.sigma = sigma

    def __call__(self, x):
        """
        应用高斯模糊

        Args:
            x (PIL.Image): 输入图像

        Returns:
            PIL.Image: 模糊后的图像
        """
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_contrastive_transforms(image_size=224, strength='medium'):
    """
    获取对比学习变换

    Args:
        image_size (int): 图像尺寸
        strength (str): 变换强度

    Returns:
        TwoCropsTransform: 双重裁剪变换
    """
    if strength == 'weak':
        base_transform = transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif strength == 'medium':
        base_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif strength == 'strong':
        base_transform = transforms.Compose([
            transforms.Resize((image_size + 64, image_size + 64)),
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported strength: {strength}")

    return TwoCropsTransform(base_transform)


class MixUp:
    """
    MixUp数据增强
    """

    def __init__(self, alpha=0.2):
        """
        初始化MixUp

        Args:
            alpha (float): Beta分布参数
        """
        self.alpha = alpha

    def __call__(self, batch):
        """
        应用MixUp

        Args:
            batch (dict): 批数据

        Returns:
            dict: MixUp后的批数据
        """
        if self.alpha <= 0:
            return batch

        batch_size = batch['images'].size(0)

        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机排列索引
        indices = torch.randperm(batch_size)

        # 混合图像
        mixed_images = lam * batch['images'] + (1 - lam) * batch['images'][indices]

        # 混合标签
        mixed_labels = lam * batch['labels'] + (1 - lam) * batch['labels'][indices]

        # 更新批数据
        batch['images'] = mixed_images
        batch['labels'] = mixed_labels
        batch['mixup_lambda'] = lam
        batch['mixup_indices'] = indices

        return batch


class CutMix:
    """
    CutMix数据增强
    """

    def __init__(self, alpha=1.0):
        """
        初始化CutMix

        Args:
            alpha (float): Beta分布参数
        """
        self.alpha = alpha

    def __call__(self, batch):
        """
        应用CutMix

        Args:
            batch (dict): 批数据

        Returns:
            dict: CutMix后的批数据
        """
        if self.alpha <= 0:
            return batch

        batch_size = batch['images'].size(0)
        _, _, H, W = batch['images'].shape

        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机排列索引
        indices = torch.randperm(batch_size)

        # 计算裁剪区域
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机选择裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 应用CutMix
        mixed_images = batch['images'].clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch['images'][indices, :, bby1:bby2, bbx1:bbx2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # 混合标签
        mixed_labels = lam * batch['labels'] + (1 - lam) * batch['labels'][indices]

        # 更新批数据
        batch['images'] = mixed_images
        batch['labels'] = mixed_labels
        batch['cutmix_lambda'] = lam
        batch['cutmix_indices'] = indices

        return batch