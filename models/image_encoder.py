"""
图像编码器模块
使用预训练的ResNet或Vision Transformer进行图像特征提取
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class ImageEncoder(nn.Module):
    """
    图像编码器，基于预训练的CNN模型
    """

    def __init__(self, backbone='resnet50', output_dim=512, pretrained=True, freeze_backbone=False):
        """
        初始化图像编码器

        Args:
            backbone (str): 骨干网络类型 ('resnet50', 'resnet101', 'vit_b_16')
            output_dim (int): 输出特征维度
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结骨干网络参数
        """
        super(ImageEncoder, self).__init__()

        self.backbone_name = backbone
        self.output_dim = output_dim

        # 选择骨干网络
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 移除最后的分类层
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 移除最后的分类层
        elif backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            backbone_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()  # 移除分类头
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 冻结骨干网络参数（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 投影层：将骨干网络输出映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

        # 全局平均池化（用于CNN特征）
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        """
        前向传播

        Args:
            images (torch.Tensor): 输入图像，形状为 [batch_size, 3, H, W]

        Returns:
            torch.Tensor: 图像特征，形状为 [batch_size, output_dim]
        """
        # 骨干网络特征提取
        features = self.backbone(images)

        # 处理不同骨干网络的输出
        if self.backbone_name in ['resnet50', 'resnet101']:
            # CNN输出需要全局平均池化
            features = self.global_pool(features)  # [batch_size, backbone_dim, 1, 1]
            features = features.flatten(1)  # [batch_size, backbone_dim]
        elif self.backbone_name == 'vit_b_16':
            # ViT输出已经是 [batch_size, backbone_dim]
            pass

        # 投影到目标维度
        features = self.projection(features)  # [batch_size, output_dim]

        # 层归一化
        features = self.layer_norm(features)

        return features


class ImageTransforms:
    """
    图像预处理变换
    """

    def __init__(self, image_size=224, is_training=True):
        """
        初始化图像变换

        Args:
            image_size (int): 图像尺寸
            is_training (bool): 是否为训练模式
        """
        self.image_size = image_size
        self.is_training = is_training

        if is_training:
            # 训练时的数据增强
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时的变换
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, image):
        """
        应用变换

        Args:
            image (PIL.Image): 输入图像

        Returns:
            torch.Tensor: 变换后的图像张量
        """
        return self.transform(image)