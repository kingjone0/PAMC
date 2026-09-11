# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/24 9:23
@Auth ： 康锦程
"""
import logging
import os
import pickle

import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []
    n_classes = len(np.unique(y_train))

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = [unq_cnt[np.argwhere(unq == i)][0, 0] if i in unq else 0 for i in range(n_classes)]
        net_cls_counts.append(tmp)
    return net_cls_counts


def partition_data(partition, n_train, client_num_in_total, labels, alpha, logger=None, save_path=None, load_path=None):
    """
    参数:
        partition: 分区方法 ('dir' 或 'pat')
        n_train: 训练样本总数
        client_num_in_total: 客户端数量
        labels: 训练标签数组
        alpha: 分区参数 (狄利克雷分布的alpha)
        save_path: 分区结果保存路径 (可选)
        load_path: 分区结果加载路径 (可选)
        logger: 日志记录器 (可选)

    返回:
        net_dataidx_map: 客户端数据索引映射
        traindata_cls_counts: 客户端类别分布统计（每个客户端每个类别的样本数）
        data_distributions: 客户端数据分布（每个客户端内每个类别的比例）
    """
    # 如果提供了加载路径且文件存在，则加载分区结果
    if load_path and os.path.exists(load_path):
        if logger:
            logger.info(f"Loading partition data from {load_path}")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data['net_dataidx_map'], data['traindata_cls_counts'], data['data_distributions']

    n_classes = len(np.unique(labels))
    ensure_class_coverage = False
    net_dataidx_map = {j: [] for j in range(client_num_in_total)}
    all_indices = set(range(n_train))  # 所有样本索引

    # 设置随机种子（可选）
    # random.seed(0)
    # np.random.seed(1)

    if partition == 'pat':
        n_client = client_num_in_total
        n_cls = n_classes

        # 计算每个客户端的期望样本数
        n_data_per_clnt = n_train / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0.2, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * n_train).astype(int)

        # 创建类别先验分布
        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            selected_cls = random.sample(range(n_cls), int(alpha))
            cls_priors[i][selected_cls] = 1.0 / alpha

        # 按类别组织索引
        idx_list = [np.where(labels == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]

        # 分配样本
        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = np.cumsum(cls_priors[curr_clnt])
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    # 重置该类别的可用样本数
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

        # 打乱每个客户端的数据
        for j in range(n_client):
            np.random.shuffle(net_dataidx_map[j])

    elif partition == 'dir':
        n_client = client_num_in_total

        # 生成狄利克雷分布
        client_distributions = np.random.dirichlet(
            np.repeat(alpha, n_classes),
            size=n_client
        )

        # 按类别分配样本
        for cls in range(n_classes):
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            if len(cls_indices) == 0:
                continue

            # 计算分配比例
            allocations = (client_distributions[:, cls] * len(cls_indices)).astype(int)
            residual = len(cls_indices) - allocations.sum()

            # 处理剩余样本
            if residual > 0:
                expand_probs = client_distributions[:, cls] / client_distributions[:, cls].sum()
                expand_clients = np.random.choice(n_client, residual, p=expand_probs)
                np.add.at(allocations, expand_clients, 1)

            # 分配样本给客户端
            ptr = 0
            for cid in range(n_client):
                if ptr >= len(cls_indices):
                    break
                end = ptr + allocations[cid]
                net_dataidx_map[cid].extend(cls_indices[ptr:end].tolist())
                ptr = end

        # 最小样本保护机制
        min_samples = 100
        for cid in range(n_client):
            current_data = net_dataidx_map[cid]
            if len(current_data) < min_samples:
                owned = set(current_data)
                candidates = list(all_indices - owned)
                needed = min_samples - len(current_data)
                if len(candidates) >= needed:
                    selected = np.random.choice(candidates, needed, replace=False)
                    net_dataidx_map[cid].extend(selected.tolist())
                    all_indices -= set(selected)
                elif logger:
                    logger.warning(f"Client {cid} cannot reach min_samples, only {len(current_data)} samples")

        # 类别覆盖检查
        if ensure_class_coverage:
            covered_classes = set()
            for cid in range(n_client):
                client_labels = labels[net_dataidx_map[cid]]
                covered_classes.update(np.unique(client_labels))

            missing_classes = set(range(n_classes)) - covered_classes
            if missing_classes:
                for cls in missing_classes:
                    cid = np.random.randint(n_client)
                    cls_indices = np.where(labels == cls)[0]
                    if len(cls_indices) > 0:
                        selected = np.random.choice(cls_indices, 1)
                        net_dataidx_map[cid].append(selected[0])
                        all_indices.remove(selected[0])
                        if logger:
                            logger.info(f"Force client {cid} to cover class {cls}")
    elif partition == 'noniid':
        min_size = 0
        min_require_size = 10
        K = int(labels.max() + 1)
        n_client = client_num_in_total
        n_cls = n_classes

        N = n_train
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_client)]
            for k in range(K):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_client))
                proportions = np.array(
                    [p * (len(idx_j) < N / n_client) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_client):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        class_dis = np.zeros((n_client, K))

        for j in range(n_client):
            for m in range(K):
                class_dis[j, m] = int((np.array(labels[idx_batch[j]]) == m).sum())

    # 打乱每个客户端的数据
    for cid in range(client_num_in_total):
        np.random.shuffle(net_dataidx_map[cid])

    # 使用record_net_data_stats函数统计训练数据的类别分布
    traindata_cls_counts = record_net_data_stats(labels, net_dataidx_map)

    # 计算每个客户端的数据分布（每个类别的比例）
    data_distributions = []
    for counts in traindata_cls_counts:
        total = sum(counts)
        if total > 0:
            data_distributions.append(np.array(counts) / total)
        else:
            data_distributions.append(np.zeros_like(counts))

    # 保存分区结果
    if save_path:
        if logger:
            logger.info(f"Saving partition data to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'net_dataidx_map': net_dataidx_map,
                'traindata_cls_counts': traindata_cls_counts,
                'data_distributions': data_distributions
            }, f)

    return net_dataidx_map, traindata_cls_counts, data_distributions

