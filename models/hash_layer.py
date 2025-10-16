"""
哈希层模块
将特征映射到哈希码空间，确保输出在-1到1之间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashLayer(nn.Module):
    """
    哈希层，将特征映射到哈希码
    """

    def __init__(self, input_dim, hash_dim, activation='tanh', temperature=1.0):
        """
        初始化哈希层

        Args:
            input_dim (int): 输入特征维度
            hash_dim (int): 哈希码维度（哈希位数）
            activation (str): 激活函数类型 ('tanh', 'sigmoid', 'gumbel')
            temperature (float): Gumbel softmax温度参数（仅在activation='gumbel'时使用）
        """
        super(HashLayer, self).__init__()

        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.activation = activation
        self.temperature = temperature

        # 哈希投影层
        self.hash_fc = nn.Sequential(
            nn.Linear(input_dim, hash_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hash_dim * 2, hash_dim)
        )

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hash_dim)

    def forward(self, features, hard=False):
        """
        前向传播

        Args:
            features (torch.Tensor): 输入特征，形状为 [batch_size, input_dim]
            hard (bool): 是否使用硬量化（仅在训练时为False，推理时为True）

        Returns:
            torch.Tensor: 哈希码，形状为 [batch_size, hash_dim]，值在-1到1之间
        """
        # 线性投影
        hash_logits = self.hash_fc(features)  # [batch_size, hash_dim]

        # 批归一化
        hash_logits = self.batch_norm(hash_logits)

        # 应用激活函数
        if self.activation == 'tanh':
            # 使用tanh激活，输出范围为(-1, 1)
            hash_codes = torch.tanh(hash_logits)

        elif self.activation == 'sigmoid':
            # 使用sigmoid激活，然后映射到(-1, 1)
            hash_codes = torch.sigmoid(hash_logits) * 2 - 1

        elif self.activation == 'gumbel':
            # 使用Gumbel Softmax进行可微分的二值化
            # 将logits转换为二分类问题
            binary_logits = torch.stack([hash_logits, -hash_logits], dim=-1)  # [batch_size, hash_dim, 2]

            if self.training and not hard:
                # 训练时使用软量化
                gumbel_softmax = F.gumbel_softmax(binary_logits, tau=self.temperature, hard=False, dim=-1)
                hash_codes = gumbel_softmax[:, :, 0] * 2 - 1  # 映射到(-1, 1)
            else:
                # 推理时使用硬量化
                hash_codes = torch.sign(hash_logits)

        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        # 如果需要硬量化（推理时）
        if hard and self.activation != 'gumbel':
            hash_codes = torch.sign(hash_codes)

        return hash_codes

    def get_binary_codes(self, features):
        """
        获取二进制哈希码（用于检索）

        Args:
            features (torch.Tensor): 输入特征

        Returns:
            torch.Tensor: 二进制哈希码，值为-1或1
        """
        with torch.no_grad():
            hash_logits = self.hash_fc(features)
            hash_logits = self.batch_norm(hash_logits)
            binary_codes = torch.sign(hash_logits)
            return binary_codes


class AdaptiveHashLayer(nn.Module):
    """
    自适应哈希层，可以根据训练过程调整量化强度
    """

    def __init__(self, input_dim, hash_dim, quantization_weight=1.0):
        """
        初始化自适应哈希层

        Args:
            input_dim (int): 输入特征维度
            hash_dim (int): 哈希码维度
            quantization_weight (float): 量化损失权重
        """
        super(AdaptiveHashLayer, self).__init__()

        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.quantization_weight = quantization_weight

        # 哈希投影层
        self.hash_fc = nn.Linear(input_dim, hash_dim)

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hash_dim)

    def forward(self, features):
        """
        前向传播

        Args:
            features (torch.Tensor): 输入特征

        Returns:
            tuple: (连续哈希码, 二进制哈希码, 量化损失)
        """
        # 线性投影
        hash_logits = self.hash_fc(features)
        hash_logits = self.batch_norm(hash_logits)

        # 连续哈希码（用于训练）
        continuous_codes = torch.tanh(hash_logits)

        # 二进制哈希码（用于推理）
        binary_codes = torch.sign(hash_logits)

        # 量化损失：鼓励连续码接近二进制码
        quantization_loss = torch.mean((continuous_codes - binary_codes.detach()) ** 2)

        return continuous_codes, binary_codes, quantization_loss