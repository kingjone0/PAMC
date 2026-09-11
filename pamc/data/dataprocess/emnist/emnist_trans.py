# -*- coding: utf-8 -*-
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST

from .data_loader import partition_data


EMNIST_SPLIT = "balanced"
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


class EMNIST_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img.astype("uint8"), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def record_part(train_cls_counts, val_cls_counts, logger):
    for net_i in range(len(train_cls_counts)):
        train_dist = train_cls_counts[net_i]
        val_dist = val_cls_counts[net_i]
        train_str = ", ".join(f"{count:4d}" for count in train_dist)
        val_str = ", ".join(f"{count:4d}" for count in val_dist)
        logger.info(f"Client {net_i:2d} Partition - Train: [{train_str}], Val: [{val_str}]")


def emnist_dataset_read(data_dir, batch_size, client_num_in_total, partition, alpha, logger, save_path=None, load_path=None):
    EMNIST.url = EMNIST_URL

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = EMNIST(data_dir, split=EMNIST_SPLIT, train=True, download=True)
    test_dataset = EMNIST(data_dir, split=EMNIST_SPLIT, train=False, download=True)

    train_image = train_dataset.data.numpy()
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data.numpy()
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    num_classes = len(np.unique(train_label))

    net_dataidx_map, _, data_distributions = partition_data(
        partition=partition,
        n_train=n_train,
        client_num_in_total=client_num_in_total,
        labels=train_label,
        alpha=alpha,
        logger=logger,
        save_path=save_path,
        load_path=load_path,
    )

    train_dataloaders = []
    val_dataloaders = []
    data_local_num_dict = {}
    train_counts_per_client = []
    val_counts_per_client = []

    for i in range(client_num_in_total):
        client_idxs = np.array(net_dataidx_map[i])
        np.random.shuffle(client_idxs)

        split_point = int(0.8 * len(client_idxs))
        train_idxs = client_idxs[:split_point]
        val_idxs = client_idxs[split_point:]

        data_local_num_dict[i] = len(train_idxs)

        train_labels = train_label[train_idxs]
        unq_train, unq_cnt_train = np.unique(train_labels, return_counts=True)
        train_cls_counts = [unq_cnt_train[unq_train == c][0] if c in unq_train else 0 for c in range(num_classes)]
        train_counts_per_client.append(train_cls_counts)

        val_labels = train_label[val_idxs]
        unq_val, unq_cnt_val = np.unique(val_labels, return_counts=True)
        val_cls_counts = [unq_cnt_val[unq_val == c][0] if c in unq_val else 0 for c in range(num_classes)]
        val_counts_per_client.append(val_cls_counts)

        train_dataset_client = EMNIST_Truncated(
            data=train_image[train_idxs],
            labels=train_label[train_idxs],
            transform=transform_train,
        )
        train_loader = DataLoader(
            dataset=train_dataset_client,
            batch_size=batch_size,
            shuffle=True,
        )
        train_dataloaders.append(train_loader)

        val_dataset_client = EMNIST_Truncated(
            data=train_image[val_idxs],
            labels=train_label[val_idxs],
            transform=transform_test,
        )
        val_loader = DataLoader(
            dataset=val_dataset_client,
            batch_size=batch_size,
            shuffle=False,
        )
        val_dataloaders.append(val_loader)

        logger.info(f"Client {i}: Train samples = {len(train_idxs)}, Val samples = {len(val_idxs)}")

    test_dataset = EMNIST_Truncated(
        data=test_image,
        labels=test_label,
        transform=transform_test,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    logger.info(f"Test samples = {len(test_loader.dataset)}")

    if logger:
        logger.info("Detailed data distribution per client:")
        record_part(train_counts_per_client, val_counts_per_client, logger)

    return train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions
