"""
哈希检索评估指标模块
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score
from typing import Dict, List, Tuple, Optional
import time


def hamming_distance(hash1, hash2):
    """
    计算汉明距离

    Args:
        hash1 (torch.Tensor): 哈希码1，形状为 [N, hash_dim]
        hash2 (torch.Tensor): 哈希码2，形状为 [M, hash_dim]

    Returns:
        torch.Tensor: 汉明距离矩阵，形状为 [N, M]
    """
    # 确保哈希码为二进制（-1或1）
    hash1_binary = torch.sign(hash1)
    hash2_binary = torch.sign(hash2)

    # 计算汉明距离
    # 汉明距离 = (hash_dim - 内积) / 2
    inner_product = torch.matmul(hash1_binary, hash2_binary.t())
    hamming_dist = (hash1_binary.size(1) - inner_product) / 2

    return hamming_dist


def euclidean_distance(feat1, feat2):
    """
    计算欧几里得距离

    Args:
        feat1 (torch.Tensor): 特征1
        feat2 (torch.Tensor): 特征2

    Returns:
        torch.Tensor: 欧几里得距离矩阵
    """
    # L2归一化
    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)

    # 计算距离
    dist_matrix = torch.cdist(feat1_norm, feat2_norm, p=2)

    return dist_matrix


def cosine_similarity(feat1, feat2):
    """
    计算余弦相似度

    Args:
        feat1 (torch.Tensor): 特征1
        feat2 (torch.Tensor): 特征2

    Returns:
        torch.Tensor: 余弦相似度矩阵
    """
    # L2归一化
    feat1_norm = torch.nn.functional.normalize(feat1, p=2, dim=1)
    feat2_norm = torch.nn.functional.normalize(feat2, p=2, dim=1)

    # 计算余弦相似度
    similarity = torch.matmul(feat1_norm, feat2_norm.t())

    return similarity


def calculate_map(query_labels, gallery_labels, distances, top_k=None):
    """
    计算平均精度均值 (mAP)

    Args:
        query_labels (torch.Tensor): 查询标签，形状为 [N]
        gallery_labels (torch.Tensor): 数据库标签，形状为 [M]
        distances (torch.Tensor): 距离矩阵，形状为 [N, M]
        top_k (int): 只考虑前k个结果，None表示考虑所有

    Returns:
        float: mAP值
    """
    num_queries = query_labels.size(0)

    # 按距离排序（升序）
    _, indices = torch.sort(distances, dim=1)

    aps = []

    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[indices[i]]

        # 找到相关的样本（标签相同）
        relevant = (sorted_gallery_labels == query_label).float()

        if top_k is not None:
            relevant = relevant[:top_k]

        # 计算平均精度
        if relevant.sum() > 0:
            ap = calculate_average_precision(relevant.cpu().numpy())
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def calculate_average_precision(relevant):
    """
    计算单个查询的平均精度

    Args:
        relevant (np.ndarray): 相关性数组，1表示相关，0表示不相关

    Returns:
        float: 平均精度
    """
    if np.sum(relevant) == 0:
        return 0.0

    # 计算精度曲线
    precisions = []
    num_relevant = 0

    for i, rel in enumerate(relevant):
        if rel == 1:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def calculate_precision_at_k(query_labels, gallery_labels, distances, k_values):
    """
    计算Precision@K

    Args:
        query_labels (torch.Tensor): 查询标签
        gallery_labels (torch.Tensor): 数据库标签
        distances (torch.Tensor): 距离矩阵
        k_values (List[int]): K值列表

    Returns:
        Dict[int, float]: 各个K值对应的Precision@K
    """
    num_queries = query_labels.size(0)

    # 按距离排序
    _, indices = torch.sort(distances, dim=1)

    precision_at_k = {k: [] for k in k_values}

    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[indices[i]]

        for k in k_values:
            if k <= sorted_gallery_labels.size(0):
                top_k_labels = sorted_gallery_labels[:k]
                relevant = (top_k_labels == query_label).float()
                precision = relevant.sum().item() / k
                precision_at_k[k].append(precision)

    # 计算平均值
    result = {}
    for k in k_values:
        if precision_at_k[k]:
            result[k] = np.mean(precision_at_k[k])
        else:
            result[k] = 0.0

    return result


def calculate_recall_at_k(query_labels, gallery_labels, distances, k_values):
    """
    计算Recall@K

    Args:
        query_labels (torch.Tensor): 查询标签
        gallery_labels (torch.Tensor): 数据库标签
        distances (torch.Tensor): 距离矩阵
        k_values (List[int]): K值列表

    Returns:
        Dict[int, float]: 各个K值对应的Recall@K
    """
    num_queries = query_labels.size(0)

    # 按距离排序
    _, indices = torch.sort(distances, dim=1)

    recall_at_k = {k: [] for k in k_values}

    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[indices[i]]

        # 计算总的相关样本数
        total_relevant = (gallery_labels == query_label).sum().item()

        for k in k_values:
            if k <= sorted_gallery_labels.size(0) and total_relevant > 0:
                top_k_labels = sorted_gallery_labels[:k]
                relevant_retrieved = (top_k_labels == query_label).sum().item()
                recall = relevant_retrieved / total_relevant
                recall_at_k[k].append(recall)

    # 计算平均值
    result = {}
    for k in k_values:
        if recall_at_k[k]:
            result[k] = np.mean(recall_at_k[k])
        else:
            result[k] = 0.0

    return result


def calculate_ndcg_at_k(query_labels, gallery_labels, distances, k_values):
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        query_labels (torch.Tensor): 查询标签
        gallery_labels (torch.Tensor): 数据库标签
        distances (torch.Tensor): 距离矩阵
        k_values (List[int]): K值列表

    Returns:
        Dict[int, float]: 各个K值对应的NDCG@K
    """
    num_queries = query_labels.size(0)

    # 按距离排序
    _, indices = torch.sort(distances, dim=1)

    ndcg_at_k = {k: [] for k in k_values}

    for i in range(num_queries):
        query_label = query_labels[i]
        sorted_gallery_labels = gallery_labels[indices[i]]

        for k in k_values:
            if k <= sorted_gallery_labels.size(0):
                top_k_labels = sorted_gallery_labels[:k]

                # 计算DCG
                dcg = 0.0
                for j, label in enumerate(top_k_labels):
                    relevance = 1.0 if label == query_label else 0.0
                    dcg += relevance / np.log2(j + 2)  # j+2 because log2(1)=0

                # 计算IDCG (理想DCG)
                num_relevant = (gallery_labels == query_label).sum().item()
                idcg = 0.0
                for j in range(min(k, num_relevant)):
                    idcg += 1.0 / np.log2(j + 2)

                # 计算NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_at_k[k].append(ndcg)

    # 计算平均值
    result = {}
    for k in k_values:
        if ndcg_at_k[k]:
            result[k] = np.mean(ndcg_at_k[k])
        else:
            result[k] = 0.0

    return result


class HashRetrievalMetrics:
    """
    哈希检索评估指标类
    """

    def __init__(self, k_values=[1, 5, 10, 20, 50], distance_metric='hamming'):
        """
        初始化评估指标

        Args:
            k_values (List[int]): 评估的K值列表
            distance_metric (str): 距离度量方式 ('hamming', 'euclidean', 'cosine')
        """
        self.k_values = k_values
        self.distance_metric = distance_metric

    def compute_distance_matrix(self, query_features, gallery_features):
        """
        计算距离矩阵

        Args:
            query_features (torch.Tensor): 查询特征
            gallery_features (torch.Tensor): 数据库特征

        Returns:
            torch.Tensor: 距离矩阵
        """
        if self.distance_metric == 'hamming':
            return hamming_distance(query_features, gallery_features)
        elif self.distance_metric == 'euclidean':
            return euclidean_distance(query_features, gallery_features)
        elif self.distance_metric == 'cosine':
            # 对于余弦相似度，返回负值作为距离（相似度越高，距离越小）
            return -cosine_similarity(query_features, gallery_features)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def evaluate(self, query_features, gallery_features, query_labels, gallery_labels):
        """
        评估检索性能

        Args:
            query_features (torch.Tensor): 查询特征
            gallery_features (torch.Tensor): 数据库特征
            query_labels (torch.Tensor): 查询标签
            gallery_labels (torch.Tensor): 数据库标签

        Returns:
            Dict: 评估结果
        """
        # 计算距离矩阵
        distances = self.compute_distance_matrix(query_features, gallery_features)

        # 计算各种指标
        results = {}

        # mAP
        map_score = calculate_map(query_labels, gallery_labels, distances)
        results['mAP'] = map_score

        # Precision@K
        precision_at_k = calculate_precision_at_k(
            query_labels, gallery_labels, distances, self.k_values
        )
        for k, precision in precision_at_k.items():
            results[f'Precision@{k}'] = precision

        # Recall@K
        recall_at_k = calculate_recall_at_k(
            query_labels, gallery_labels, distances, self.k_values
        )
        for k, recall in recall_at_k.items():
            results[f'Recall@{k}'] = recall

        # NDCG@K
        ndcg_at_k = calculate_ndcg_at_k(
            query_labels, gallery_labels, distances, self.k_values
        )
        for k, ndcg in ndcg_at_k.items():
            results[f'NDCG@{k}'] = ndcg

        return results

    def evaluate_cross_modal(self,
                             text_features, image_features,
                             text_labels, image_labels):
        """
        评估跨模态检索性能

        Args:
            text_features (torch.Tensor): 文本特征
            image_features (torch.Tensor): 图像特征
            text_labels (torch.Tensor): 文本标签
            image_labels (torch.Tensor): 图像标签

        Returns:
            Dict: 跨模态评估结果
        """
        results = {}

        # 文本查询图像 (Text-to-Image)
        t2i_results = self.evaluate(text_features, image_features, text_labels, image_labels)
        for key, value in t2i_results.items():
            results[f'T2I_{key}'] = value

        # 图像查询文本 (Image-to-Text)
        i2t_results = self.evaluate(image_features, text_features, image_labels, text_labels)
        for key, value in i2t_results.items():
            results[f'I2T_{key}'] = value

        # 平均性能
        avg_results = {}
        for key in t2i_results.keys():
            avg_value = (t2i_results[key] + i2t_results[key]) / 2
            avg_results[f'Avg_{key}'] = avg_value

        results.update(avg_results)

        return results


def compute_retrieval_metrics_batch(query_hashes, gallery_hashes, query_labels, gallery_labels,
                                    k_values=[1, 5, 10, 20, 50], batch_size=1000):
    """
    批量计算检索指标（用于大规模数据）

    Args:
        query_hashes (torch.Tensor): 查询哈希码
        gallery_hashes (torch.Tensor): 数据库哈希码
        query_labels (torch.Tensor): 查询标签
        gallery_labels (torch.Tensor): 数据库标签
        k_values (List[int]): K值列表
        batch_size (int): 批处理大小

    Returns:
        Dict: 评估结果
    """
    num_queries = query_hashes.size(0)

    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_aps = []

    # 分批处理查询
    for start_idx in range(0, num_queries, batch_size):
        end_idx = min(start_idx + batch_size, num_queries)

        batch_query_hashes = query_hashes[start_idx:end_idx]
        batch_query_labels = query_labels[start_idx:end_idx]

        # 计算距离矩阵
        distances = hamming_distance(batch_query_hashes, gallery_hashes)

        # 计算指标
        batch_map = calculate_map(batch_query_labels, gallery_labels, distances)
        all_aps.append(batch_map)

        batch_precisions = calculate_precision_at_k(
            batch_query_labels, gallery_labels, distances, k_values
        )
        batch_recalls = calculate_recall_at_k(
            batch_query_labels, gallery_labels, distances, k_values
        )

        for k in k_values:
            all_precisions[k].append(batch_precisions[k])
            all_recalls[k].append(batch_recalls[k])

    # 汇总结果
    results = {
        'mAP': np.mean(all_aps)
    }

    for k in k_values:
        results[f'Precision@{k}'] = np.mean(all_precisions[k])
        results[f'Recall@{k}'] = np.mean(all_recalls[k])

    return results