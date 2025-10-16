#!/usr/bin/env python3
"""
跨模态哈希检索推理脚本

使用示例:
    python inference.py --model_path checkpoints/best_model.pth --query_text "A cat sitting on a chair"
    python inference.py --model_path checkpoints/best_model.pth --query_image path/to/image.jpg
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cross_modal_hash import CrossModalHashModel
from models.text_encoder import TextTokenizer
from data.transforms import get_transforms
from evaluation.metrics import hamming_distance, cosine_similarity
from utils.utils import setup_device


class CrossModalRetriever:
    """
    跨模态检索器
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化检索器

        Args:
            model_path (str): 模型路径
            device (str): 计算设备
        """
        self.device = setup_device(device)

        # 加载模型
        self.model, self.config = self.load_model(model_path)
        self.model.eval()

        # 创建分词器和图像变换
        self.tokenizer = TextTokenizer(
            model_name=self.config.get('text_model', 'bert-base-uncased'),
            max_length=self.config.get('max_text_length', 128)
        )

        self.image_transform = get_transforms(
            image_size=self.config.get('image_size', 224),
            is_training=False
        )

        # 数据库（用于检索）
        self.text_database = []
        self.image_database = []
        self.text_hashes = None
        self.image_hashes = None
        self.labels = None

    def load_model(self, model_path: str) -> Tuple[CrossModalHashModel, Dict]:
        """
        加载模型

        Args:
            model_path (str): 模型路径

        Returns:
            tuple: (模型, 配置)
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})

        # 创建模型
        model = CrossModalHashModel(
            hash_dim=config.get('hash_dim', 64),
            feature_dim=config.get('feature_dim', 512),
            text_model=config.get('text_model', 'bert-base-uncased'),
            image_backbone=config.get('image_backbone', 'resnet50'),
            hash_activation=config.get('hash_activation', 'tanh'),
            use_adaptive_hash=config.get('use_adaptive_hash', False),
            temperature=config.get('temperature', 1.0)
        )

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        print(f"Model loaded from {model_path}")
        print(f"Hash dimension: {config.get('hash_dim', 64)}")

        return model, config

    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本

        Args:
            text (str): 输入文本

        Returns:
            tuple: (特征, 哈希码)
        """
        # 分词
        text_inputs = self.tokenizer.encode(text, return_tensors='pt')
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # 编码
        with torch.no_grad():
            outputs = self.model.encode_text(**text_inputs)
            features = outputs['features']
            hash_codes = outputs.get('hash_codes', outputs.get('continuous_hash'))

            # 二值化哈希码
            binary_hash = torch.sign(hash_codes)

        return features, binary_hash

    def encode_image(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像

        Args:
            image_path (str): 图像路径

        Returns:
            tuple: (特征, 哈希码)
        """
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # 编码
        with torch.no_grad():
            outputs = self.model.encode_image(image_tensor)
            features = outputs['features']
            hash_codes = outputs.get('hash_codes', outputs.get('continuous_hash'))

            # 二值化哈希码
            binary_hash = torch.sign(hash_codes)

        return features, binary_hash

    def build_database(self, text_list: List[str] = None, image_paths: List[str] = None,
                       labels: List[int] = None):
        """
        构建检索数据库

        Args:
            text_list (List[str]): 文本列表
            image_paths (List[str]): 图像路径列表
            labels (List[int]): 标签列表
        """
        print("Building retrieval database...")

        text_hashes = []
        image_hashes = []

        # 编码文本
        if text_list:
            print(f"Encoding {len(text_list)} texts...")
            for text in text_list:
                _, hash_code = self.encode_text(text)
                text_hashes.append(hash_code)
            self.text_database = text_list

        # 编码图像
        if image_paths:
            print(f"Encoding {len(image_paths)} images...")
            for image_path in image_paths:
                _, hash_code = self.encode_image(image_path)
                image_hashes.append(hash_code)
            self.image_database = image_paths

        # 保存哈希码
        if text_hashes:
            self.text_hashes = torch.cat(text_hashes, dim=0)
        if image_hashes:
            self.image_hashes = torch.cat(image_hashes, dim=0)

        # 保存标签
        if labels:
            self.labels = torch.tensor(labels)

        print("Database built successfully!")

    def retrieve_by_text(self, query_text: str, top_k: int = 10,
                         target_modality: str = 'image') -> List[Tuple[int, float, str]]:
        """
        使用文本查询

        Args:
            query_text (str): 查询文本
            top_k (int): 返回前k个结果
            target_modality (str): 目标模态 ('image' 或 'text')

        Returns:
            List[Tuple[int, float, str]]: 检索结果 (索引, 距离, 内容)
        """
        # 编码查询文本
        _, query_hash = self.encode_text(query_text)

        # 选择目标数据库
        if target_modality == 'image':
            if self.image_hashes is None:
                raise ValueError("Image database is empty")
            target_hashes = self.image_hashes
            target_database = self.image_database
        else:
            if self.text_hashes is None:
                raise ValueError("Text database is empty")
            target_hashes = self.text_hashes
            target_database = self.text_database

        # 计算汉明距离
        distances = hamming_distance(query_hash, target_hashes).squeeze(0)

        # 排序并获取top-k
        _, indices = torch.sort(distances)
        top_indices = indices[:top_k]

        # 构建结果
        results = []
        for idx in top_indices:
            idx_val = idx.item()
            distance = distances[idx_val].item()
            content = target_database[idx_val] if target_database else f"Item {idx_val}"
            results.append((idx_val, distance, content))

        return results

    def retrieve_by_image(self, query_image_path: str, top_k: int = 10,
                          target_modality: str = 'text') -> List[Tuple[int, float, str]]:
        """
        使用图像查询

        Args:
            query_image_path (str): 查询图像路径
            top_k (int): 返回前k个结果
            target_modality (str): 目标模态 ('text' 或 'image')

        Returns:
            List[Tuple[int, float, str]]: 检索结果 (索引, 距离, 内容)
        """
        # 编码查询图像
        _, query_hash = self.encode_image(query_image_path)

        # 选择目标数据库
        if target_modality == 'text':
            if self.text_hashes is None:
                raise ValueError("Text database is empty")
            target_hashes = self.text_hashes
            target_database = self.text_database
        else:
            if self.image_hashes is None:
                raise ValueError("Image database is empty")
            target_hashes = self.image_hashes
            target_database = self.image_database

        # 计算汉明距离
        distances = hamming_distance(query_hash, target_hashes).squeeze(0)

        # 排序并获取top-k
        _, indices = torch.sort(distances)
        top_indices = indices[:top_k]

        # 构建结果
        results = []
        for idx in top_indices:
            idx_val = idx.item()
            distance = distances[idx_val].item()
            content = target_database[idx_val] if target_database else f"Item {idx_val}"
            results.append((idx_val, distance, content))

        return results

    def print_results(self, results: List[Tuple[int, float, str]], query: str):
        """
        打印检索结果

        Args:
            results (List[Tuple[int, float, str]]): 检索结果
            query (str): 查询内容
        """
        print(f"\nQuery: {query}")
        print("=" * 80)
        print(f"{'Rank':<5} {'Distance':<10} {'Content'}")
        print("-" * 80)

        for rank, (idx, distance, content) in enumerate(results, 1):
            # 截断长文本
            if len(content) > 60:
                content = content[:57] + "..."
            print(f"{rank:<5} {distance:<10.2f} {content}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Cross-Modal Hash Retrieval Inference')

    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--query_text', type=str, help='查询文本')
    parser.add_argument('--query_image', type=str, help='查询图像路径')
    parser.add_argument('--database_texts', type=str, nargs='+', help='数据库文本列表')
    parser.add_argument('--database_images', type=str, nargs='+', help='数据库图像路径列表')
    parser.add_argument('--top_k', type=int, default=10, help='返回前k个结果')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')

    return parser.parse_args()


def main():
    """
    主推理函数
    """
    args = parse_args()

    # 创建检索器
    retriever = CrossModalRetriever(args.model_path, args.device)

    # 构建数据库
    if args.database_texts or args.database_images:
        retriever.build_database(
            text_list=args.database_texts,
            image_paths=args.database_images
        )
    else:
        # 使用示例数据库
        example_texts = [
            "A cat sitting on a chair",
            "A dog running in the park",
            "A bird flying in the sky",
            "A car driving on the road",
            "A person walking on the street"
        ]
        retriever.build_database(text_list=example_texts)

    # 执行查询
    if args.query_text:
        # 文本查询图像
        print(f"Text-to-Image retrieval:")
        results = retriever.retrieve_by_text(
            args.query_text,
            top_k=args.top_k,
            target_modality='image' if retriever.image_database else 'text'
        )
        retriever.print_results(results, args.query_text)

    if args.query_image:
        # 图像查询文本
        print(f"Image-to-Text retrieval:")
        results = retriever.retrieve_by_image(
            args.query_image,
            top_k=args.top_k,
            target_modality='text' if retriever.text_database else 'image'
        )
        retriever.print_results(results, f"Image: {args.query_image}")

    if not args.query_text and not args.query_image:
        print("Please provide either --query_text or --query_image")


if __name__ == '__main__':
    main()