"""
跨模态数据集处理模块
支持COCO、Flickr30K等数据集
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np


class CrossModalDataset(Dataset):
    """
    通用跨模态数据集基类
    """

    def __init__(self,
                 image_dir: str,
                 annotations_file: str,
                 tokenizer,
                 image_transform=None,
                 max_text_length: int = 128,
                 split: str = 'train'):
        """
        初始化数据集

        Args:
            image_dir (str): 图像目录路径
            annotations_file (str): 标注文件路径
            tokenizer: 文本分词器
            image_transform: 图像变换
            max_text_length (int): 最大文本长度
            split (str): 数据集分割 ('train', 'val', 'test')
        """
        self.image_dir = image_dir
        self.annotations_file = annotations_file
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_text_length = max_text_length
        self.split = split

        # 加载数据
        self.data = self.load_annotations()

    def load_annotations(self):
        """
        加载标注数据，子类需要实现
        """
        raise NotImplementedError("Subclasses must implement load_annotations method")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据项

        Returns:
            dict: 包含图像、文本、标签等信息
        """
        item = self.data[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, item['image_file'])
        image = Image.open(image_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        # 处理文本
        text = item['caption']
        text_inputs = self.tokenizer.encode(
            text,
            # max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 移除batch维度
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return {
            'image': image,
            'text_inputs': text_inputs,
            'text_raw': text,
            'image_id': item.get('image_id', idx),
            'caption_id': item.get('caption_id', idx),
            'label': item.get('label', 0)  # 用于监督学习的标签
        }


class COCODataset(CrossModalDataset):
    """
    COCO数据集
    """

    def load_annotations(self):
        """
        加载COCO标注数据
        """
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)

        # 构建图像ID到文件名的映射
        image_id_to_filename = {}
        for img_info in coco_data['images']:
            image_id_to_filename[img_info['id']] = img_info['file_name']

        # 构建数据列表
        data = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in image_id_to_filename:
                data.append({
                    'image_file': image_id_to_filename[image_id],
                    'caption': ann['caption'],
                    'image_id': image_id,
                    'caption_id': ann['id'],
                    'label': image_id  # 使用image_id作为标签，同一图像的不同caption有相同标签
                })

        return data


class Flickr30KDataset(CrossModalDataset):
    """
    Flickr30K数据集
    """

    def load_annotations(self):
        """
        加载Flickr30K标注数据
        """
        data = []

        # 假设标注文件是CSV格式，包含image_name和caption列
        if self.annotations_file.endswith('.csv'):
            df = pd.read_csv(self.annotations_file)
            for idx, row in df.iterrows():
                data.append({
                    'image_file': row['image_name'],
                    'caption': row['caption'],
                    'image_id': row.get('image_id', idx),
                    'caption_id': idx,
                    'label': row.get('image_id', idx)
                })

        # 如果是JSON格式
        elif self.annotations_file.endswith('.json'):
            with open(self.annotations_file, 'r') as f:
                flickr_data = json.load(f)

            for item in flickr_data:
                data.append({
                    'image_file': item['image_name'],
                    'caption': item['caption'],
                    'image_id': item.get('image_id', len(data)),
                    'caption_id': len(data),
                    'label': item.get('image_id', len(data))
                })

        return data


class SyntheticDataset(CrossModalDataset):
    """
    合成数据集，用于测试和演示
    """

    def __init__(self,
                 num_samples: int = 1000,
                 num_classes: int = 10,
                 tokenizer=None,
                 image_transform=None,
                 max_text_length: int = 128):
        """
        初始化合成数据集

        Args:
            num_samples (int): 样本数量
            num_classes (int): 类别数量
            tokenizer: 文本分词器
            image_transform: 图像变换
            max_text_length (int): 最大文本长度
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_text_length = max_text_length

        # 生成合成数据
        self.data = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """
        生成合成数据
        """
        data = []

        # 预定义的描述模板
        templates = [
            "A photo of a {}",
            "An image showing a {}",
            "A picture of a {}",
            "A {} in the scene",
            "A beautiful {}",
            "A colorful {}",
            "A large {}",
            "A small {}"
        ]

        # 预定义的类别名称
        class_names = [
                          "cat", "dog", "bird", "car", "bicycle",
                          "person", "tree", "flower", "house", "mountain"
                      ][:self.num_classes]

        for i in range(self.num_samples):
            # 随机选择类别
            class_idx = np.random.randint(0, self.num_classes)
            class_name = class_names[class_idx]

            # 随机选择描述模板
            template = np.random.choice(templates)
            caption = template.format(class_name)

            data.append({
                'image_file': f"synthetic_{i:06d}.jpg",  # 虚拟文件名
                'caption': caption,
                'image_id': i,
                'caption_id': i,
                'label': class_idx,
                'synthetic': True
            })

        return data

    def __getitem__(self, idx):
        """
        获取合成数据项
        """
        item = self.data[idx]

        # 生成随机图像 (3, 224, 224)
        image = torch.randn(3, 224, 224)

        if self.image_transform:
            # 如果有变换，先转换为PIL图像再应用变换
            image_pil = Image.fromarray(
                (torch.clamp(image.permute(1, 2, 0) * 0.5 + 0.5, 0, 1) * 255).numpy().astype(np.uint8))
            image = self.image_transform(image_pil)

        # 处理文本
        text = item['caption']
        if self.tokenizer:
            text_inputs = self.tokenizer.encode(
                text,
                # max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # 移除batch维度
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        else:
            # 如果没有tokenizer，返回原始文本
            text_inputs = {'text': text}

        return {
            'image': image,
            'text_inputs': text_inputs,
            'text_raw': text,
            'image_id': item['image_id'],
            'caption_id': item['caption_id'],
            'label': item['label']
        }


class PairwiseDataset(Dataset):
    """
    配对数据集，用于对比学习
    """

    def __init__(self, base_dataset, negative_sampling_ratio=1.0):
        """
        初始化配对数据集

        Args:
            base_dataset: 基础数据集
            negative_sampling_ratio (float): 负样本采样比例
        """
        self.base_dataset = base_dataset
        self.negative_sampling_ratio = negative_sampling_ratio

        # 构建标签到样本索引的映射
        self.label_to_indices = {}
        for idx, item in enumerate(base_dataset.data):
            label = item['label']
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        获取配对数据
        """
        # 获取锚点样本
        anchor_data = self.base_dataset[idx]
        anchor_label = anchor_data['label']

        # 采样正样本（同一标签的其他样本）
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        if positive_indices:
            pos_idx = np.random.choice(positive_indices)
            positive_data = self.base_dataset[pos_idx]
        else:
            # 如果没有其他正样本，使用自己
            positive_data = anchor_data

        # 采样负样本（不同标签的样本）
        negative_labels = [label for label in self.labels if label != anchor_label]
        if negative_labels:
            neg_label = np.random.choice(negative_labels)
            neg_idx = np.random.choice(self.label_to_indices[neg_label])
            negative_data = self.base_dataset[neg_idx]
        else:
            # 如果没有负样本，使用随机样本
            neg_idx = np.random.randint(0, len(self.base_dataset))
            negative_data = self.base_dataset[neg_idx]

        return {
            'anchor': anchor_data,
            'positive': positive_data,
            'negative': negative_data,
            'label': anchor_label
        }