# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/5 16:07
@Auth ： 康锦程
"""
import copy
import gc

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pamc.models.trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.args = args

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data, device, args=None, round=100, logger=None, feature_hook=None):
        """
        本地训练：
        - 与原始版本基本一致，只是增加了 feature_hook，用于在训练循环中顺带统计特征
        - 如果提供 feature_hook(feats, labels)，会在每个 batch 的 forward 后调用一次
        """
        if args is None:
            args = self.args

        model = self.model
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr * (args.lr_decay ** round),
                momentum=args.momentum,
                weight_decay=args.wd,
            )
        else:
            # 你可以按需扩展 adam 等，这里保持和原代码一致，仅实现 sgd
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr * (args.lr_decay ** round),
                momentum=args.momentum,
                weight_decay=args.wd,
            )

        for epoch in range(args.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()

                # ==== 关键改动：显式拆成 base + classifier，便于重用中间特征 ====
                feats = None
                if hasattr(model, "base") and hasattr(model, "classifier"):
                    # 标准结构：model.base -> features, model.classifier -> logits
                    feats = model.base(x)                    # [B, feat_dim]
                    log_probs = model.classifier(feats)      # [B, num_classes]
                else:
                    # 回退：如果模型没有 base/classifier，就直接 forward
                    log_probs = model.forward(x)

                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()

                # 训练同时：顺带调用 feature_hook，更新客户端侧的原型统计
                if feature_hook is not None and feats is not None:
                    with torch.no_grad():
                        feature_hook(feats.detach(), labels.detach())

                epoch_loss.append(loss.item())

            if logger is not None:
                logger.info(
                    'Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss))
                )

    @torch.jit.script
    def compute_model_similarity_batch(flattened_params_1: torch.Tensor, flattened_params_list_2: torch.Tensor):
        """
        使用余弦相似度批量计算模型相似性，进行优化以提升计算速度。
        flattened_params_1: 当前客户端的展平模型参数，尺寸为 [param_dim]
        flattened_params_list_2: 其他客户端的展平模型参数列表，尺寸为 [num_neighbors, param_dim]
        """
        # 计算批量的余弦相似性（保持在 GPU 上）
        similarities = F.cosine_similarity(flattened_params_1.unsqueeze(0), flattened_params_list_2, dim=1)
        return similarities

    def compute_model_similarity_batch_wrapper(self, model_params_1, model_params_list_2):
        device = torch.device("cuda:" + str(self.args.gpu))

        with torch.no_grad():
            # 将模型参数展平成一维张量，并传递到 GPU
            flattened_params_1 = weight_flatten_all(model_params_1).to(device)
            flattened_params_list_2 = torch.stack([weight_flatten_all(params) for params in model_params_list_2]).to(
                device)

            # 调用 JIT 编译的函数进行计算
            similarities = self.compute_model_similarity_batch(flattened_params_1, flattened_params_list_2)

            # 删除中间结果以释放内存，避免频繁调用 empty_cache
            del flattened_params_1, flattened_params_list_2
            gc.collect()
            torch.cuda.empty_cache()

            return similarities

    @torch.jit.script
    def compute_data_complementarity_batch(subspace_1: torch.Tensor, subspace_list_2: torch.Tensor):
        """
        使用主角度的方法批量计算数据互补性，进行优化以提升计算速度。
        subspace_1: 当前客户端的特征子空间，尺寸为 [num_components, feature_dim]
        subspace_list_2: 其他客户端的特征子空间列表，每个为 [num_components, feature_dim]
        """
        # 将 subspace_1 扩展为与 subspace_list_2 相同的批次大小
        subspace_1_expanded = subspace_1.unsqueeze(0).expand(subspace_list_2.size(0), -1,
                                                             -1)  # [num_neighbors, num_components, feature_dim]

        # 计算每个子空间对之间的余弦相似度矩阵
        norm_A = torch.norm(subspace_1_expanded, dim=2, keepdim=True)  # [num_neighbors, num_components, 1]
        norm_B = torch.norm(subspace_list_2, dim=2, keepdim=True)  # [num_neighbors, num_components, 1]
        dot_product = torch.bmm(subspace_list_2,
                                subspace_1_expanded.transpose(1, 2))  # [num_neighbors, num_components, num_components]
        cosine_matrix = dot_product / (
                    norm_B * norm_A.transpose(1, 2))  # [num_neighbors, num_components, num_components]

        # 使用 torch.topk 找出每一行的最大值索引（保持在 GPU 上处理）
        cos_phi_values, _ = torch.topk(cosine_matrix.flatten(start_dim=1), k=cosine_matrix.shape[1], dim=1)

        # 计算主角度并得到互补性（保持在 GPU 上）
        phi = torch.arccos(torch.clamp(cos_phi_values, -1, 1))
        complementarity = torch.cos((1 / cosine_matrix.shape[1]) * torch.sum(phi, dim=1))

        return complementarity

    def compute_data_complementarity_batch_wrapper(self, subspace_1, subspace_list_2):
        device = torch.device("cuda:" + str(self.args.gpu))

        with torch.no_grad():
            # 将 subspace_1 和 subspace_list_2 转移到 GPU
            subspace_1 = subspace_1.to(device)  # [num_components, feature_dim]
            subspace_list_2 = torch.stack(
                [subspace.to(device) for subspace in subspace_list_2])  # [num_neighbors, num_components, feature_dim]

            # 调用 JIT 编译的函数进行计算
            complementarity_tensor = self.compute_data_complementarity_batch(subspace_1, subspace_list_2)

            # 将结果转为列表（如果需要在 Python 中使用）
            complementarity_list = complementarity_tensor.cpu().tolist()

            # 清理不再使用的变量
            del subspace_1, subspace_list_2
            gc.collect()
            torch.cuda.empty_cache()

            return complementarity_list

    @torch.jit.script
    def compute_logits_complementarity_batch(logits_1: torch.Tensor, logits_list_2: torch.Tensor):
        # 扩展 logits_1 为 [num_neighbors, num_classes]
        logits_1_expanded = logits_1.unsqueeze(0).expand(logits_list_2.size(0), -1)

        # 添加极小值并重新归一化
        logits_1_expanded = (logits_1_expanded + 1e-8) / (logits_1_expanded.sum(dim=1, keepdim=True) + 1e-8)
        logits_list_2 = (logits_list_2 + 1e-8) / (logits_list_2.sum(dim=1, keepdim=True) + 1e-8)

        # 计算对称的JS散度
        M = 0.5 * (logits_1_expanded + logits_list_2)
        kl_pm = F.kl_div(torch.log(M), logits_1_expanded, reduction='none', log_target=False).sum(dim=1)
        kl_qm = F.kl_div(torch.log(M), logits_list_2, reduction='none', log_target=False).sum(dim=1)
        js_divergence = 0.5 * (kl_pm + kl_qm)  # [num_neighbors]

        return js_divergence

    def compute_logits_complementarity_batch_wrapper(self, logits_1, logits_list_2):
        device = torch.device("cuda:" + str(self.args.gpu))
        logits_1 = logits_1.to(device)
        logits_list_2 = torch.stack([logits.to(device) for logits in logits_list_2])

        # 转换为概率分布
        logits_1 = F.softmax(logits_1, dim=-1)
        logits_list_2 = F.softmax(logits_list_2, dim=-1)

        # 计算对称的JS散度
        js_divergence = self.compute_logits_complementarity_batch(logits_1, logits_list_2)

        # 使用非线性变换增强区分度
        # 方案1: 使用指数变换
        complementarities = torch.exp(-js_divergence * 10)  # 将JS散度转换为相似度

        # 方案2: 使用sigmoid缩放
        # complementarities = torch.sigmoid((1 - js_divergence) * 5)

        # 方案3: 使用幂变换
        # complementarities = torch.pow(1 - js_divergence, 0.5)

        return complementarities

    def test(self, test_data, device, args, data_distribution=None):
        model = self.model.to(device)
        model.eval()

        generalized_total = 0
        generalized_correct = 0
        correct_conf_sum = 0.0
        incorrect_conf_sum = 0.0
        incorrect_count = 0

        # 类别统计（仅当需要个性化准确率时使用）
        if data_distribution is not None:
            class_correct = np.zeros(args.num_classes)
            class_total = np.zeros(args.num_classes)
            data_distribution = np.array(data_distribution)

        with torch.no_grad():
            for x, labels in test_data:
                x, labels = x.to(device), labels.to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)

                correct_mask = (preds == labels)
                batch_correct = correct_mask.sum().item()
                batch_size = labels.size(0)

                generalized_total += batch_size
                generalized_correct += batch_correct

                correct_conf_sum += confidences[correct_mask].sum().item()
                incorrect_conf_sum += confidences[~correct_mask].sum().item()
                incorrect_count += (batch_size - batch_correct)

                if data_distribution is not None:
                    for i in range(batch_size):
                        label = labels[i].item()
                        if label < args.num_classes:
                            class_total[label] += 1
                            if correct_mask[i]:
                                class_correct[label] += 1

        generalized_acc = generalized_correct / generalized_total if generalized_total > 0 else 0.0
        avg_correct_conf = correct_conf_sum / generalized_correct if generalized_correct > 0 else 0.0
        avg_incorrect_conf = incorrect_conf_sum / incorrect_count if incorrect_count > 0 else 0.0

        personalized_acc = 0.0
        if data_distribution is not None:
            class_acc = np.zeros(args.num_classes)
            for i in range(args.num_classes):
                if class_total[i] > 0:
                    class_acc[i] = class_correct[i] / class_total[i]

            personalized_acc = np.sum(class_acc * data_distribution)

        # 返回四个关键指标
        return {
            'generalized_acc': generalized_acc,
            'personalized_acc': personalized_acc,
            'avg_correct_conf': avg_correct_conf,
            'avg_incorrect_conf': avg_incorrect_conf
        }

    def test_after_aggregated(self, test_data, device):
        model = self.model.to(device)
        model.eval()

        generalized_total = 0
        generalized_correct = 0

        with torch.no_grad():
            for x, labels in test_data:
                x, labels = x.to(device), labels.to(device)

                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                confidences, preds = torch.max(probs, dim=1)

                correct_mask = (preds == labels)
                batch_correct = correct_mask.sum().item()
                batch_size = labels.size(0)

                generalized_total += batch_size
                generalized_correct += batch_correct


        generalized_acc = generalized_correct / generalized_total if generalized_total > 0 else 0.0

        return generalized_acc

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    # 新增：基于本地数据集评估平均交叉熵损失
    def eval_loss(self, data_loader, device):
        model = self.model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for x, labels in data_loader:
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                loss = criterion(logits, labels.long())
                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
        avg_loss = total_loss / max(total_samples, 1)
        return avg_loss

    # 新增：给定参数字典评估损失（便于“聚合前后”的立即评估）
    def eval_loss_for_params(self, model_parameters, data_loader, device):
        self.set_model_params(model_parameters)
        return self.eval_loss(data_loader, device)


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

