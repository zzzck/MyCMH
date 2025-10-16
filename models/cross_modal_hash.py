"""
跨模态哈希检索模型
整合文本编码器、图像编码器和哈希层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .hash_layer import HashLayer, AdaptiveHashLayer


class CrossModalHashModel(nn.Module):
    """
    跨模态哈希检索模型
    """

    def __init__(self,
                 hash_dim=64,
                 feature_dim=512,
                 text_model='bert-base-uncased',
                 image_backbone='resnet50',
                 hash_activation='tanh',
                 use_adaptive_hash=False,
                 temperature=1.0):
        """
        初始化跨模态哈希模型

        Args:
            hash_dim (int): 哈希码维度
            feature_dim (int): 特征维度
            text_model (str): 文本模型名称
            image_backbone (str): 图像骨干网络
            hash_activation (str): 哈希激活函数
            use_adaptive_hash (bool): 是否使用自适应哈希层
            temperature (float): Gumbel softmax温度
        """
        super(CrossModalHashModel, self).__init__()

        self.hash_dim = hash_dim
        self.feature_dim = feature_dim
        self.use_adaptive_hash = use_adaptive_hash

        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=feature_dim
        )

        # 图像编码器
        self.image_encoder = ImageEncoder(
            backbone=image_backbone,
            output_dim=feature_dim
        )

        # 哈希层
        if use_adaptive_hash:
            self.text_hash_layer = AdaptiveHashLayer(feature_dim, hash_dim)
            self.image_hash_layer = AdaptiveHashLayer(feature_dim, hash_dim)
        else:
            self.text_hash_layer = HashLayer(feature_dim, hash_dim, hash_activation, temperature)
            self.image_hash_layer = HashLayer(feature_dim, hash_dim, hash_activation, temperature)

        # 模态对齐层（可选）
        self.text_projection = nn.Linear(feature_dim, feature_dim)
        self.image_projection = nn.Linear(feature_dim, feature_dim)

    def encode_text(self, input_ids, attention_mask=None, token_type_ids=None, return_hash=True):
        """
        编码文本

        Args:
            input_ids (torch.Tensor): 输入token ids
            attention_mask (torch.Tensor): 注意力掩码
            token_type_ids (torch.Tensor): token类型ids
            return_hash (bool): 是否返回哈希码

        Returns:
            dict: 包含特征和哈希码的字典
        """
        # 文本特征提取
        text_features = self.text_encoder(input_ids, attention_mask, token_type_ids)

        # 特征投影
        text_features = self.text_projection(text_features)

        result = {'features': text_features}

        if return_hash:
            if self.use_adaptive_hash:
                continuous_hash, binary_hash, quant_loss = self.text_hash_layer(text_features)
                result.update({
                    'continuous_hash': continuous_hash,
                    'binary_hash': binary_hash,
                    'quantization_loss': quant_loss
                })
            else:
                hash_codes = self.text_hash_layer(text_features)
                result['hash_codes'] = hash_codes

        return result

    def encode_image(self, images, return_hash=True):
        """
        编码图像

        Args:
            images (torch.Tensor): 输入图像
            return_hash (bool): 是否返回哈希码

        Returns:
            dict: 包含特征和哈希码的字典
        """
        # 图像特征提取
        image_features = self.image_encoder(images)

        # 特征投影
        image_features = self.image_projection(image_features)

        result = {'features': image_features}

        if return_hash:
            if self.use_adaptive_hash:
                continuous_hash, binary_hash, quant_loss = self.image_hash_layer(image_features)
                result.update({
                    'continuous_hash': continuous_hash,
                    'binary_hash': binary_hash,
                    'quantization_loss': quant_loss
                })
            else:
                hash_codes = self.image_hash_layer(image_features)
                result['hash_codes'] = hash_codes

        return result

    def forward(self, text_inputs=None, images=None):
        """
        前向传播

        Args:
            text_inputs (dict): 文本输入，包含input_ids, attention_mask等
            images (torch.Tensor): 图像输入

        Returns:
            dict: 模型输出
        """
        results = {}

        if text_inputs is not None:
            text_results = self.encode_text(**text_inputs)
            results['text'] = text_results

        if images is not None:
            image_results = self.encode_image(images)
            results['image'] = image_results

        return results

    def get_hash_codes(self, text_inputs=None, images=None, hard=True):
        """
        获取哈希码（用于检索）

        Args:
            text_inputs (dict): 文本输入
            images (torch.Tensor): 图像输入
            hard (bool): 是否使用硬量化

        Returns:
            dict: 哈希码
        """
        with torch.no_grad():
            results = {}

            if text_inputs is not None:
                text_features = self.text_encoder(**text_inputs)
                text_features = self.text_projection(text_features)

                if self.use_adaptive_hash:
                    _, text_hash, _ = self.text_hash_layer(text_features)
                else:
                    text_hash = self.text_hash_layer(text_features, hard=hard)

                results['text_hash'] = text_hash

            if images is not None:
                image_features = self.image_encoder(images)
                image_features = self.image_projection(image_features)

                if self.use_adaptive_hash:
                    _, image_hash, _ = self.image_hash_layer(image_features)
                else:
                    image_hash = self.image_hash_layer(image_features, hard=hard)

                results['image_hash'] = image_hash

            return results


class CrossModalHashLoss(nn.Module):
    """
    跨模态哈希损失函数
    """

    def __init__(self,
                 margin=0.2,
                 lambda_quant=0.1,
                 lambda_balance=0.01,
                 temperature=0.07):
        """
        初始化损失函数

        Args:
            margin (float): 对比学习边界
            lambda_quant (float): 量化损失权重
            lambda_balance (float): 平衡损失权重
            temperature (float): 对比学习温度
        """
        super(CrossModalHashLoss, self).__init__()

        self.margin = margin
        self.lambda_quant = lambda_quant
        self.lambda_balance = lambda_balance
        self.temperature = temperature

    def contrastive_loss(self, text_features, image_features, labels):
        """
        对比学习损失

        Args:
            text_features (torch.Tensor): 文本特征或哈希码
            image_features (torch.Tensor): 图像特征或哈希码
            labels (torch.Tensor): 标签，1表示匹配，0表示不匹配

        Returns:
            torch.Tensor: 对比损失
        """
        # 计算相似度矩阵
        text_norm = F.normalize(text_features, p=2, dim=1)
        image_norm = F.normalize(image_features, p=2, dim=1)

        similarity = torch.matmul(text_norm, image_norm.t()) / self.temperature

        # 构建标签矩阵
        batch_size = text_features.size(0)
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        labels_matrix = labels_matrix.float()

        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity)

        # 正样本损失
        pos_mask = labels_matrix
        pos_sim = similarity * pos_mask
        pos_loss = -torch.log(torch.sum(torch.exp(pos_sim) * pos_mask, dim=1) /
                              torch.sum(exp_sim, dim=1) + 1e-8)

        return torch.mean(pos_loss)

    def quantization_loss(self, hash_codes):
        """
        量化损失，鼓励哈希码接近-1或1

        Args:
            hash_codes (torch.Tensor): 哈希码

        Returns:
            torch.Tensor: 量化损失
        """
        # 计算每个哈希位与最近的二进制值(-1或1)的距离
        quantization_error = torch.mean((hash_codes - torch.sign(hash_codes)) ** 2)
        return quantization_error

    def balance_loss(self, hash_codes):
        """
        平衡损失，鼓励每个哈希位的均值接近0

        Args:
            hash_codes (torch.Tensor): 哈希码

        Returns:
            torch.Tensor: 平衡损失
        """
        # 计算每个哈希位的均值
        bit_means = torch.mean(hash_codes, dim=0)
        balance_error = torch.mean(bit_means ** 2)
        return balance_error

    def forward(self, text_outputs, image_outputs, labels):
        """
        计算总损失

        Args:
            text_outputs (dict): 文本编码输出
            image_outputs (dict): 图像编码输出
            labels (torch.Tensor): 标签

        Returns:
            dict: 损失字典
        """
        losses = {}

        # 特征对比损失
        if 'features' in text_outputs and 'features' in image_outputs:
            feature_contrastive = self.contrastive_loss(
                text_outputs['features'],
                image_outputs['features'],
                labels
            )
            losses['feature_contrastive'] = feature_contrastive

        # 哈希码对比损失
        text_hash_key = 'continuous_hash' if 'continuous_hash' in text_outputs else 'hash_codes'
        image_hash_key = 'continuous_hash' if 'continuous_hash' in image_outputs else 'hash_codes'

        if text_hash_key in text_outputs and image_hash_key in image_outputs:
            hash_contrastive = self.contrastive_loss(
                text_outputs[text_hash_key],
                image_outputs[image_hash_key],
                labels
            )
            losses['hash_contrastive'] = hash_contrastive

        # 量化损失
        quantization_losses = []
        if 'quantization_loss' in text_outputs:
            quantization_losses.append(text_outputs['quantization_loss'])
        if 'quantization_loss' in image_outputs:
            quantization_losses.append(image_outputs['quantization_loss'])

        if text_hash_key in text_outputs:
            quantization_losses.append(self.quantization_loss(text_outputs[text_hash_key]))
        if image_hash_key in image_outputs:
            quantization_losses.append(self.quantization_loss(image_outputs[image_hash_key]))

        if quantization_losses:
            losses['quantization'] = torch.mean(torch.stack(quantization_losses))

        # 平衡损失
        balance_losses = []
        if text_hash_key in text_outputs:
            balance_losses.append(self.balance_loss(text_outputs[text_hash_key]))
        if image_hash_key in image_outputs:
            balance_losses.append(self.balance_loss(image_outputs[image_hash_key]))

        if balance_losses:
            losses['balance'] = torch.mean(torch.stack(balance_losses))

        # 总损失
        total_loss = 0
        if 'feature_contrastive' in losses:
            total_loss += losses['feature_contrastive']
        if 'hash_contrastive' in losses:
            total_loss += losses['hash_contrastive']
        if 'quantization' in losses:
            total_loss += self.lambda_quant * losses['quantization']
        if 'balance' in losses:
            total_loss += self.lambda_balance * losses['balance']

        losses['total'] = total_loss

        return losses