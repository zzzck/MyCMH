"""
跨模态哈希检索评估器
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging
from .metrics import HashRetrievalMetrics, compute_retrieval_metrics_batch


class CrossModalEvaluator:
    """
    跨模态哈希检索评估器
    """

    def __init__(self,
                 k_values=[1, 5, 10, 20, 50],
                 distance_metrics=['hamming', 'cosine'],
                 batch_size=1000,
                 device='cuda'):
        """
        初始化评估器

        Args:
            k_values (List[int]): 评估的K值列表
            distance_metrics (List[str]): 距离度量方式列表
            batch_size (int): 批处理大小
            device (str): 设备
        """
        self.k_values = k_values
        self.distance_metrics = distance_metrics
        self.batch_size = batch_size
        self.device = device

        # 为每种距离度量创建评估器
        self.metrics_calculators = {}
        for metric in distance_metrics:
            self.metrics_calculators[metric] = HashRetrievalMetrics(
                k_values=k_values,
                distance_metric=metric
            )

    def extract_features(self, model, dataloader, return_labels=True):
        """
        提取特征和哈希码

        Args:
            model: 跨模态哈希模型
            dataloader: 数据加载器
            return_labels (bool): 是否返回标签

        Returns:
            Dict: 提取的特征、哈希码和标签
        """
        model.eval()

        text_features = []
        image_features = []
        text_hashes = []
        image_hashes = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                images = batch['images'].to(self.device)
                text_inputs = {k: v.to(self.device) for k, v in batch['text_inputs'].items()}
                batch_labels = batch['labels'].to(self.device)

                # 编码文本
                text_outputs = model.encode_text(**text_inputs)
                text_feat = text_outputs['features']
                text_hash = text_outputs.get('hash_codes', text_outputs.get('continuous_hash'))

                # 编码图像
                image_outputs = model.encode_image(images)
                image_feat = image_outputs['features']
                image_hash = image_outputs.get('hash_codes', image_outputs.get('continuous_hash'))

                # 收集结果
                text_features.append(text_feat.cpu())
                image_features.append(image_feat.cpu())
                text_hashes.append(text_hash.cpu())
                image_hashes.append(image_hash.cpu())

                if return_labels:
                    labels.append(batch_labels.cpu())

        # 拼接所有批次
        results = {
            'text_features': torch.cat(text_features, dim=0),
            'image_features': torch.cat(image_features, dim=0),
            'text_hashes': torch.cat(text_hashes, dim=0),
            'image_hashes': torch.cat(image_hashes, dim=0)
        }

        if return_labels:
            results['labels'] = torch.cat(labels, dim=0)

        return results

    def evaluate_model(self, model, test_dataloader, split_ratio=0.5):
        """
        评估模型性能

        Args:
            model: 跨模态哈希模型
            test_dataloader: 测试数据加载器
            split_ratio (float): 查询集和数据库集的分割比例

        Returns:
            Dict: 评估结果
        """
        print("Extracting features and hash codes...")
        start_time = time.time()

        # 提取所有特征和哈希码
        extracted_data = self.extract_features(model, test_dataloader)

        text_features = extracted_data['text_features']
        image_features = extracted_data['image_features']
        text_hashes = extracted_data['text_hashes']
        image_hashes = extracted_data['image_hashes']
        labels = extracted_data['labels']

        extract_time = time.time() - start_time
        print(f"Feature extraction completed in {extract_time:.2f}s")

        # 分割查询集和数据库集
        num_samples = labels.size(0)
        num_query = int(num_samples * split_ratio)

        # 随机打乱索引
        indices = torch.randperm(num_samples)
        query_indices = indices[:num_query]
        gallery_indices = indices[num_query:]

        # 分割数据
        query_text_features = text_features[query_indices]
        query_image_features = image_features[query_indices]
        query_text_hashes = text_hashes[query_indices]
        query_image_hashes = image_hashes[query_indices]
        query_labels = labels[query_indices]

        gallery_text_features = text_features[gallery_indices]
        gallery_image_features = image_features[gallery_indices]
        gallery_text_hashes = text_hashes[gallery_indices]
        gallery_image_hashes = image_hashes[gallery_indices]
        gallery_labels = labels[gallery_indices]

        print(f"Query set size: {num_query}, Gallery set size: {num_samples - num_query}")

        # 评估不同距离度量下的性能
        all_results = {}

        for metric_name in self.distance_metrics:
            print(f"\nEvaluating with {metric_name} distance...")
            metric_start_time = time.time()

            calculator = self.metrics_calculators[metric_name]

            if metric_name == 'hamming':
                # 使用哈希码进行评估
                results = calculator.evaluate_cross_modal(
                    query_text_hashes, query_image_hashes,
                    gallery_text_hashes, gallery_image_hashes,
                    # query_labels, gallery_labels
                )
            else:
                # 使用特征进行评估
                results = calculator.evaluate_cross_modal(
                    query_text_features, query_image_features,
                    gallery_text_features, gallery_image_features,
                    # query_labels, gallery_labels
                )

            # 添加距离度量前缀
            for key, value in results.items():
                all_results[f"{metric_name}_{key}"] = value

            metric_time = time.time() - metric_start_time
            print(f"{metric_name} evaluation completed in {metric_time:.2f}s")

        # 添加一些额外的统计信息
        all_results['num_query'] = num_query
        all_results['num_gallery'] = num_samples - num_query
        all_results['hash_dim'] = text_hashes.size(1)
        all_results['feature_dim'] = text_features.size(1)
        all_results['total_time'] = time.time() - start_time

        return all_results

    def evaluate_cross_modal_separate(self,
                                      text_features, image_features,
                                      text_hashes, image_hashes,
                                      text_labels, image_labels):
        """
        分别评估文本到图像和图像到文本的检索性能

        Args:
            text_features (torch.Tensor): 文本特征
            image_features (torch.Tensor): 图像特征
            text_hashes (torch.Tensor): 文本哈希码
            image_hashes (torch.Tensor): 图像哈希码
            text_labels (torch.Tensor): 文本标签
            image_labels (torch.Tensor): 图像标签

        Returns:
            Dict: 分别的评估结果
        """
        results = {}

        for metric_name in self.distance_metrics:
            calculator = self.metrics_calculators[metric_name]

            if metric_name == 'hamming':
                # 文本查询图像
                t2i_results = calculator.evaluate(
                    text_hashes, image_hashes, text_labels, image_labels
                )

                # 图像查询文本
                i2t_results = calculator.evaluate(
                    image_hashes, text_hashes, image_labels, text_labels
                )
            else:
                # 文本查询图像
                t2i_results = calculator.evaluate(
                    text_features, image_features, text_labels, image_labels
                )

                # 图像查询文本
                i2t_results = calculator.evaluate(
                    image_features, text_features, image_labels, text_labels
                )

            # 保存结果
            for key, value in t2i_results.items():
                results[f"{metric_name}_T2I_{key}"] = value

            for key, value in i2t_results.items():
                results[f"{metric_name}_I2T_{key}"] = value

            # 计算平均值
            for key in t2i_results.keys():
                avg_value = (t2i_results[key] + i2t_results[key]) / 2
                results[f"{metric_name}_Avg_{key}"] = avg_value

        return results

    def print_results(self, results, title="Evaluation Results"):
        """
        打印评估结果

        Args:
            results (Dict): 评估结果
            title (str): 标题
        """
        print(f"\n{'=' * 50}")
        print(f"{title:^50}")
        print(f"{'=' * 50}")

        # 按距离度量分组显示
        for metric in self.distance_metrics:
            print(f"\n{metric.upper()} Distance Metric:")
            print("-" * 30)

            # 显示主要指标
            main_metrics = ['mAP', 'Precision@1', 'Precision@5', 'Precision@10']

            for direction in ['T2I', 'I2T', 'Avg']:
                print(f"\n{direction} Results:")
                for metric_name in main_metrics:
                    key = f"{metric}_{direction}_{metric_name}"
                    if key in results:
                        print(f"  {metric_name}: {results[key]:.4f}")

        # 显示统计信息
        if 'num_query' in results:
            print(f"\nStatistics:")
            print(f"  Query samples: {results['num_query']}")
            print(f"  Gallery samples: {results['num_gallery']}")
            print(f"  Hash dimension: {results['hash_dim']}")
            print(f"  Feature dimension: {results['feature_dim']}")
            print(f"  Total time: {results['total_time']:.2f}s")

    def save_results(self, results, filepath):
        """
        保存评估结果到文件

        Args:
            results (Dict): 评估结果
            filepath (str): 保存路径
        """
        import json

        # 转换tensor为float
        results_to_save = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results_to_save[key] = value.item()
            else:
                results_to_save[key] = value

        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to {filepath}")


def quick_evaluate(model, test_dataloader, device='cuda', k_values=[1, 5, 10]):
    """
    快速评估函数

    Args:
        model: 跨模态哈希模型
        test_dataloader: 测试数据加载器
        device (str): 设备
        k_values (List[int]): K值列表

    Returns:
        Dict: 评估结果
    """
    evaluator = CrossModalEvaluator(
        k_values=k_values,
        distance_metrics=['hamming'],
        device=device
    )

    results = evaluator.evaluate_model(model, test_dataloader)
    evaluator.print_results(results, "Quick Evaluation")

    return results