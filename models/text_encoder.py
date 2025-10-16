"""
文本编码器模块
使用预训练的BERT模型进行文本特征提取
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    """
    文本编码器，基于BERT模型
    """

    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, output_dim=512, freeze_bert=False):
        """
        初始化文本编码器

        Args:
            model_name (str): BERT模型名称
            hidden_dim (int): BERT隐藏层维度
            output_dim (int): 输出特征维度
            freeze_bert (bool): 是否冻结BERT参数
        """
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 冻结BERT参数（可选）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 投影层：将BERT输出映射到指定维度
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播

        Args:
            input_ids (torch.Tensor): 输入token ids，形状为 [batch_size, seq_len]
            attention_mask (torch.Tensor): 注意力掩码，形状为 [batch_size, seq_len]
            token_type_ids (torch.Tensor): token类型ids，形状为 [batch_size, seq_len]

        Returns:
            torch.Tensor: 文本特征，形状为 [batch_size, output_dim]
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用[CLS]标记的输出作为句子表示
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]

        # 投影到目标维度
        features = self.projection(pooled_output)  # [batch_size, output_dim]

        # 层归一化
        features = self.layer_norm(features)

        return features


class TextTokenizer:
    """
    文本分词器包装类
    """

    def __init__(self, model_name='bert-base-uncased', max_length=128):
        """
        初始化分词器

        Args:
            model_name (str): BERT模型名称
            max_length (int): 最大序列长度
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, texts, padding=True, truncation=True, return_tensors='pt'):
        """
        编码文本

        Args:
            texts (list or str): 输入文本
            padding (bool): 是否填充
            truncation (bool): 是否截断
            return_tensors (str): 返回张量类型

        Returns:
            dict: 编码结果，包含input_ids, attention_mask等
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )