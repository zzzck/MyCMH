"""
数据加载器模块
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np


def collate_fn(batch):
    """
    自定义批处理函数

    Args:
        batch (list): 批数据列表

    Returns:
        dict: 批处理后的数据
    """
    # 分离不同类型的数据
    images = []
    text_inputs = {}
    text_raw = []
    image_ids = []
    caption_ids = []
    labels = []

    for item in batch:
        images.append(item['image'])
        text_raw.append(item['text_raw'])
        image_ids.append(item['image_id'])
        caption_ids.append(item['caption_id'])
        labels.append(item['label'])

        # 处理文本输入
        for key, value in item['text_inputs'].items():
            if key not in text_inputs:
                text_inputs[key] = []
            text_inputs[key].append(value)

    # 堆叠数据
    batch_data = {
        'images': torch.stack(images),
        'text_inputs': {key: torch.stack(values) for key, values in text_inputs.items()},
        'text_raw': text_raw,
        'image_ids': torch.tensor(image_ids),
        'caption_ids': torch.tensor(caption_ids),
        'labels': torch.tensor(labels)
    }

    return batch_data


def pairwise_collate_fn(batch):
    """
    配对数据的批处理函数

    Args:
        batch (list): 配对批数据列表

    Returns:
        dict: 批处理后的配对数据
    """
    anchors = []
    positives = []
    negatives = []
    labels = []

    for item in batch:
        anchors.append(item['anchor'])
        positives.append(item['positive'])
        negatives.append(item['negative'])
        labels.append(item['label'])

    # 使用标准collate_fn处理每组数据
    anchor_batch = collate_fn(anchors)
    positive_batch = collate_fn(positives)
    negative_batch = collate_fn(negatives)

    return {
        'anchor': anchor_batch,
        'positive': positive_batch,
        'negative': negative_batch,
        'labels': torch.tensor(labels)
    }


def create_dataloader(dataset,
                      batch_size=32,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=True,
                      drop_last=False,
                      distributed=False,
                      pairwise=False):
    """
    创建数据加载器

    Args:
        dataset: 数据集
        batch_size (int): 批大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 工作进程数
        pin_memory (bool): 是否固定内存
        drop_last (bool): 是否丢弃最后一个不完整批次
        distributed (bool): 是否使用分布式采样
        pairwise (bool): 是否使用配对数据

    Returns:
        DataLoader: 数据加载器
    """
    # 选择采样器
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # 分布式采样器已经处理了shuffle
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # 选择批处理函数
    collate_func = pairwise_collate_fn if pairwise else collate_fn

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_func
    )

    return dataloader


class BalancedBatchSampler:
    """
    平衡批采样器，确保每个批次包含多个类别
    """

    def __init__(self, dataset, batch_size, samples_per_class=2):
        """
        初始化平衡批采样器

        Args:
            dataset: 数据集
            batch_size (int): 批大小
            samples_per_class (int): 每个类别的样本数
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        # 构建类别到样本索引的映射
        self.class_to_indices = {}
        for idx, item in enumerate(dataset.data):
            label = item['label']
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        # 计算每个批次需要的类别数
        self.classes_per_batch = batch_size // samples_per_class

    def __iter__(self):
        """
        迭代器
        """
        # 随机打乱类别顺序
        shuffled_classes = np.random.permutation(self.classes)

        for i in range(0, len(shuffled_classes), self.classes_per_batch):
            batch_classes = shuffled_classes[i:i + self.classes_per_batch]
            batch_indices = []

            for cls in batch_classes:
                # 从每个类别中随机采样指定数量的样本
                class_indices = self.class_to_indices[cls]
                if len(class_indices) >= self.samples_per_class:
                    selected_indices = np.random.choice(
                        class_indices,
                        size=self.samples_per_class,
                        replace=False
                    )
                else:
                    # 如果样本不够，使用重复采样
                    selected_indices = np.random.choice(
                        class_indices,
                        size=self.samples_per_class,
                        replace=True
                    )

                batch_indices.extend(selected_indices)

            # 如果批次不够大，随机填充
            while len(batch_indices) < self.batch_size:
                random_idx = np.random.randint(0, len(self.dataset))
                batch_indices.append(random_idx)

            # 打乱批次内的顺序
            np.random.shuffle(batch_indices)

            yield batch_indices[:self.batch_size]

    def __len__(self):
        """
        返回批次数量
        """
        return len(self.classes) // self.classes_per_batch


def create_balanced_dataloader(dataset,
                               batch_size=32,
                               samples_per_class=2,
                               num_workers=4,
                               pin_memory=True):
    """
    创建平衡数据加载器

    Args:
        dataset: 数据集
        batch_size (int): 批大小
        samples_per_class (int): 每个类别的样本数
        num_workers (int): 工作进程数
        pin_memory (bool): 是否固定内存

    Returns:
        DataLoader: 平衡数据加载器
    """
    batch_sampler = BalancedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        samples_per_class=samples_per_class
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return dataloader