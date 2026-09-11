# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/22 15:38
@Auth ： 康锦程
"""
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from  .data_loader import partition_data


class Cifar_Truncated(data.Dataset):
    """统一的数据集封装类"""

    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def record_part(train_cls_counts, val_cls_counts, logger):
    """
    记录数据分区情况（使用预计算的分布统计）

    参数:
        train_cls_counts: 每个客户端训练集分布统计列表
        val_cls_counts: 每个客户端验证集分布统计列表
        logger: 日志记录器
    """
    for net_i in range(len(train_cls_counts)):
        # 直接使用预计算的分布
        train_dist = train_cls_counts[net_i]
        val_dist = val_cls_counts[net_i]

        # 格式化输出
        train_str = ', '.join(f"{count:4d}" for count in train_dist)
        val_str = ', '.join(f"{count:4d}" for count in val_dist)

        # 记录日志
        logger.info(f"Client {net_i:2d} Partition - Train: [{train_str}], Val: [{val_str}]")


def cifar10_dataset_read(data_dir, batch_size, client_num_in_total, partition, alpha, logger, save_path=None,
                       load_path=None):
    """
        加载CIFAR数据集，进行分区处理，并创建数据加载器

        参数:
            dataset (str): 数据集名称，支持 'cifar10' 或 'cifar100'
            data_dir (str): 数据集存储路径
            batch_size (int): 数据加载器的批量大小
            client_num_in_total (int): 客户端总数
            partition (str): 数据分区方法名称
            alpha (float): 数据分区参数（如Dirichlet分布的alpha值）
            logger (Logger): 日志记录器对象
            save_path (str, optional): 分区结果保存路径. 默认为 None.
            load_path (str, optional): 分区结果加载路径. 默认为 None.

        返回:
            train_dataloaders (list): 客户端训练数据加载器列表
            val_dataloaders (list): 客户端验证数据加载器列表
            test_loader (DataLoader): 测试集数据加载器
            data_local_num_dict (dict): 客户端训练数据量字典 {客户端ID: 样本数量}
            data_distributions (list): 客户端数据分布统计信息
        """
    # 1. 数据集预处理定义
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(data_dir, train=True, download=True)
    test_dataset = CIFAR10(data_dir, train=False, download=True)


    # 2. 提取数据和标签
    train_image = train_dataset.data
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]

    # 3. 数据划分
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(
        partition=partition,
        n_train=n_train,
        client_num_in_total=client_num_in_total,
        labels=train_label,
        alpha=alpha,
        logger=logger,
        save_path=save_path,
        load_path=load_path
    )

    # 4. 创建数据加载器
    train_dataloaders = []
    val_dataloaders = []
    data_local_num_dict = {}  # 客户端训练数据量字典

    # 初始化数据分布统计
    train_counts_per_client = []
    val_counts_per_client = []

    for i in range(client_num_in_total):
        # 获取当前客户端的数据索引
        client_idxs = net_dataidx_map[i]
        np.random.shuffle(client_idxs)

        # 分割训练集和验证集 (80% 训练, 20% 验证)
        split_point = int(0.8 * len(client_idxs))
        train_idxs = client_idxs[:split_point]
        val_idxs = client_idxs[split_point:]

        # 记录客户端训练数据量
        data_local_num_dict[i] = len(train_idxs)

        # 统计训练集分布
        train_labels = train_label[train_idxs]
        unq_train, unq_cnt_train = np.unique(train_labels, return_counts=True)
        train_cls_counts = [unq_cnt_train[unq_train == c][0] if c in unq_train else 0
                            for c in range(10)]
        train_counts_per_client.append(train_cls_counts)

        # 统计验证集分布
        val_labels = train_label[val_idxs]
        unq_val, unq_cnt_val = np.unique(val_labels, return_counts=True)
        val_cls_counts = [unq_cnt_val[unq_val == c][0] if c in unq_val else 0
                          for c in range(10)]
        val_counts_per_client.append(val_cls_counts)

        # 创建训练数据集和加载器
        train_dataset_client = Cifar_Truncated(
            data=train_image[train_idxs],
            labels=train_label[train_idxs],
            transform=transform_train
        )
        train_loader = DataLoader(
            dataset=train_dataset_client,
            batch_size=batch_size,
            shuffle=True
        )
        train_dataloaders.append(train_loader)

        # 创建验证数据集和加载器
        val_dataset_client = Cifar_Truncated(
            data=train_image[val_idxs],
            labels=train_label[val_idxs],
            transform=transform_test
        )
        val_loader = DataLoader(
            dataset=val_dataset_client,
            batch_size=batch_size,
            shuffle=False
        )
        val_dataloaders.append(val_loader)

        logger.info(f"Client {i}: Train samples = {len(train_idxs)}, Val samples = {len(val_idxs)}")

    # 5. 创建测试集加载器
    test_dataset = Cifar_Truncated(
        data=test_image,
        labels=test_label,
        transform=transform_test
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    logger.info(f"Test samples = {len(test_loader.dataset)}")

    # 6. 记录数据分布
    if logger:
        logger.info("Detailed data distribution per client:")
        # 这里需要实现record_part函数（根据你的实际需求）
        record_part(train_counts_per_client, val_counts_per_client, logger)

    return train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions