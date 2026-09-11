# -*- coding: utf-8 -*-
import os
import pickle
import random

import numpy as np


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []
    n_classes = len(np.unique(y_train))

    for _, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = [unq_cnt[np.argwhere(unq == i)][0, 0] if i in unq else 0 for i in range(n_classes)]
        net_cls_counts.append(tmp)
    return net_cls_counts


def partition_data(partition, n_train, client_num_in_total, labels, alpha, logger=None, save_path=None, load_path=None):
    if load_path and os.path.exists(load_path):
        if logger:
            logger.info(f"Loading partition data from {load_path}")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        return data["net_dataidx_map"], data["traindata_cls_counts"], data["data_distributions"]

    n_classes = len(np.unique(labels))
    ensure_class_coverage = False
    net_dataidx_map = {j: [] for j in range(client_num_in_total)}
    all_indices = set(range(n_train))

    if partition == "pat":
        n_client = client_num_in_total
        n_cls = n_classes
        n_data_per_clnt = n_train / n_client
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0.2, size=n_client)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * n_train).astype(int)

        cls_priors = np.zeros(shape=(n_client, n_cls))
        for i in range(n_client):
            selected_cls = random.sample(range(n_cls), int(alpha))
            cls_priors[i][selected_cls] = 1.0 / alpha

        idx_list = [np.where(labels == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_client)
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = np.cumsum(cls_priors[curr_clnt])
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = len(idx_list[cls_label])
                    continue
                cls_amount[cls_label] -= 1
                net_dataidx_map[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break

        for j in range(n_client):
            np.random.shuffle(net_dataidx_map[j])

    elif partition == "dir":
        n_client = client_num_in_total
        client_distributions = np.random.dirichlet(np.repeat(alpha, n_classes), size=n_client)

        for cls in range(n_classes):
            cls_indices = np.where(labels == cls)[0]
            np.random.shuffle(cls_indices)
            if len(cls_indices) == 0:
                continue

            allocations = (client_distributions[:, cls] * len(cls_indices)).astype(int)
            residual = len(cls_indices) - allocations.sum()

            if residual > 0:
                expand_probs = client_distributions[:, cls] / client_distributions[:, cls].sum()
                expand_clients = np.random.choice(n_client, residual, p=expand_probs)
                np.add.at(allocations, expand_clients, 1)

            ptr = 0
            for cid in range(n_client):
                if ptr >= len(cls_indices):
                    break
                end = ptr + allocations[cid]
                net_dataidx_map[cid].extend(cls_indices[ptr:end].tolist())
                ptr = end

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

    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        n_client = client_num_in_total
        n_classes = int(labels.max() + 1)
        n_total = n_train

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_client)]
            for k in range(n_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(alpha, n_client))
                proportions = np.array(
                    [p * (len(idx_j) < n_total / n_client) for p, idx_j in zip(proportions, idx_batch)]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min(len(idx_j) for idx_j in idx_batch)

        net_dataidx_map = {}
        for j in range(n_client):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    for cid in range(client_num_in_total):
        np.random.shuffle(net_dataidx_map[cid])

    traindata_cls_counts = record_net_data_stats(labels, net_dataidx_map)

    data_distributions = []
    for counts in traindata_cls_counts:
        total = sum(counts)
        if total > 0:
            data_distributions.append(np.array(counts) / total)
        else:
            data_distributions.append(np.zeros_like(counts))

    if save_path:
        if logger:
            logger.info(f"Saving partition data to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "net_dataidx_map": net_dataidx_map,
                    "traindata_cls_counts": traindata_cls_counts,
                    "data_distributions": data_distributions,
                },
                f,
            )

    return net_dataidx_map, traindata_cls_counts, data_distributions
