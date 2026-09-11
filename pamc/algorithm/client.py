import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from pamc.algorithm.rl_agent import MetricWeightPolicy, PPOAgent


def _flatten_params(params_dict) -> torch.Tensor:
    """
    将 state_dict 展平成一维向量，用于计算参数范数 / 差值范数。
    默认在 CPU 上返回，避免显存压力。
    """
    flats = []
    for v in params_dict.values():
        flats.append(v.detach().reshape(-1).cpu())
    if len(flats) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(flats, dim=0)


class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
        data_distribution,
        expected_acc,
        logger,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger = logger
        self.args = args
        self.device = device
        self.data_distribution = data_distribution
        self.model_trainer = model_trainer

        self.diag_reward_ema = 0.0
        self.diag_reward_var_ema = 0.0
        self.diag_ema_beta = getattr(self.args, "diag_ema_beta", 0.05)  # 0.05~0.1都行

        self.last_action_scores = None  # 保存 action_scores 便于诊断
        self.last_metric_weights = np.array(
            [
                float(getattr(self.args, "heuristic_sim_weight", 0.2)),
                float(getattr(self.args, "heuristic_feat_weight", 0.4)),
                float(getattr(self.args, "heuristic_logit_weight", 0.4)),
            ],
            dtype=np.float32,
        )
        self.last_metric_weights = self.last_metric_weights / (self.last_metric_weights.sum() + 1e-8)
        self.last_selected_metric_components = np.zeros(3, dtype=np.float32)
        self.last_selected_metric_score = 0.0

        # 模型输入尺寸 / batch_size（用于随机特征生成）
        if hasattr(self.args, "model"):
            if self.args.model == "cnn_cifar100":
                self.input_size = 192
                self.batch_size = 32
            elif self.args.model == "cnn_cifar10":
                self.input_size = 84
                self.batch_size = 32
            else:
                self.input_size = getattr(self.args, "input_size", 84)
                self.batch_size = getattr(self.args, "batch_size", 32)
        else:
            self.input_size = getattr(self.args, "input_size", 84)
            self.batch_size = getattr(self.args, "batch_size", 32)

        # ============ RL 策略网络初始化 ============
        self._init_rl_agent()

        # 期望精度 / EMA
        self.expected_acc = expected_acc
        self.expected_ema_eta = 0.1

        # 本地验证损失 EMA 作为“自状态”的一部分
        self.loss_ema_eta = 0.05
        try:
            self.expected_val_loss = self.model_trainer.eval_loss(
                self.local_test_data, self.device
            )
        except Exception:
            self.expected_val_loss = 1.0

        # Behavior stats for RL state (entropy, maxprob, margin, acc_delta, loss_delta)
        self.behavior_stats = np.zeros(5, dtype=np.float32)
        self.last_val_loss = float(self.expected_val_loss) if self.expected_val_loss is not None else 1.0
        self.last_val_acc = float(self.expected_acc) if self.expected_acc is not None else 0.0

        # ===== 模型参数相关的“自状态”特征 =====
        # 当前轮参数更新范数比例 ||Δθ|| / ||θ||
        self.delta_theta_norm = 0.0
        # 当前模型参数的 L2 范数
        self.param_norm = 0.0

        # 上一轮参数的展平向量（如需更复杂统计可使用）
        self.prev_flat_params = None

        # PCA & 随机特征
        self.pca_fitted = False
        self.pca_basis = None
        self.random_features = self.generate_random_features()

        # 三个邻居指标：模型相似性 / 特征互补性 / 分布互补性（logits 互补）
        self.model_similarity_list = {}
        self.feature_complementarity_list = {}
        self.logits_complementarity_list = {}
        self.neighbor_sample_num_dict = {}
        self.selection_count = {}
        self.accuracy_history = []

        # 初始化自身指标
        self.model_similarity_list[self.client_idx] = 1.0
        self.feature_complementarity_list[self.client_idx] = 0.0
        self.logits_complementarity_list[self.client_idx] = 0.0
        self.neighbor_sample_num_dict[self.client_idx] = self.local_sample_number
        self.neighbor_last_seen_round = {}
        self.neighbor_staleness_dict = {}
        self.neighbor_observed_dict = {}
        self._init_neighbor_metric_cache()

        # 通信评分（如需）
        self.communication_scores = {}

        # ---- RL 相关缓存 ----
        # 一条轨迹缓存（当前版本默认单步更新；如需滑动窗口可只改 update_rl_agent）
        self.rl_states = []
        self.rl_actions = []
        self.rl_log_probs = []
        self.rl_rewards = []
        self.rl_values = []

        # 仅统计用的 reward 历史（不参与 PPO）
        self.reward_history = []

        # 上次选择的邻居和权重
        self.last_selected_contexts = {}
        self.last_selected_alpha = {}
        self.last_selected_neighbors = []
        self.last_aggregation_weights = {}

        # warmup pointer（若需要）
        self.warmup_pointer = 0

        # ========= 原型统计缓存（由 train() 内部的 feature_hook 更新） =========
        self.proto_class_feat_sums = None
        self.proto_class_counts = None
        self.proto_global_feat_sum = None
        self.proto_global_count = 0
        self.cached_soft_logits = None

    # =========================================================
    # RL policy init and trajectory buffers
    # =========================================================
    def _init_rl_agent(self):
        """
        Initialize RL policy and agent.
        """
        # Fixed summary state: self state + neighbor metric/staleness summary + last policy summary.
        state_dim = 22
        action_dim = 3

        hidden_dim = getattr(self.args, "rl_hidden_dim", 256)
        rl_lr = getattr(self.args, "rl_lr", 3e-4)
        rl_gamma = getattr(self.args, "rl_gamma", 0.99)
        rl_clip = getattr(self.args, "rl_clip_epsilon", 0.2)

        policy_net = MetricWeightPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )
        self.policy_net = policy_net
        self.rl_agent = PPOAgent(
            policy_net=policy_net,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            lr=rl_lr,
            gamma=rl_gamma,
            clip_epsilon=rl_clip,
            update_epochs=getattr(self.args, "ppo_update_epochs", 4),
            value_coef=getattr(self.args, "ppo_value_coef", 0.5),
            entropy_coef=getattr(self.args, "ppo_entropy_coef", 0.01),
            dirichlet_scale=getattr(self.args, "ppo_dirichlet_scale", 10.0),
        )

    def _init_neighbor_metric_cache(self):
        total = getattr(self.args, "client_num_in_total", 0)
        for other_id in range(total):
            if other_id == self.client_idx:
                continue
            self.model_similarity_list.setdefault(other_id, 0.5)
            self.feature_complementarity_list.setdefault(other_id, 0.5)
            self.logits_complementarity_list.setdefault(other_id, 0.5)
            self.neighbor_sample_num_dict.setdefault(other_id, 0)
            self.neighbor_last_seen_round.setdefault(other_id, -1)
            self.neighbor_staleness_dict.setdefault(other_id, 0)
            self.neighbor_observed_dict.setdefault(other_id, 0.0)

    def update_neighbor_metric_cache(
        self,
        other_client_idx: int,
        sim: float,
        feat_comp: float,
        logit_comp: float,
        sample_num: int,
        round_idx: int,
    ):
        self.model_similarity_list[other_client_idx] = float(sim)
        self.feature_complementarity_list[other_client_idx] = float(feat_comp)
        self.logits_complementarity_list[other_client_idx] = float(logit_comp)
        self.neighbor_sample_num_dict[other_client_idx] = int(sample_num)
        self.neighbor_last_seen_round[other_client_idx] = int(round_idx)
        self.neighbor_staleness_dict[other_client_idx] = 0
        self.neighbor_observed_dict[other_client_idx] = 1.0

    def advance_neighbor_staleness(self, round_idx: int):
        for other_id in range(self.args.client_num_in_total):
            if other_id == self.client_idx:
                continue
            last_seen = self.neighbor_last_seen_round.get(other_id, -1)
            if last_seen < 0:
                self.neighbor_staleness_dict[other_id] = int(round_idx + 1)
            else:
                self.neighbor_staleness_dict[other_id] = max(0, int(round_idx - last_seen))

    def update_selected_metric_summary(self, selected_neighbors):
        if selected_neighbors:
            sims = [self.model_similarity_list.get(j, 0.0) for j in selected_neighbors]
            feats = [self.feature_complementarity_list.get(j, 0.0) for j in selected_neighbors]
            logits = [self.logits_complementarity_list.get(j, 0.0) for j in selected_neighbors]
            self.last_selected_metric_components = np.array(
                [float(np.mean(sims)), float(np.mean(feats)), float(np.mean(logits))],
                dtype=np.float32,
            )
            self.last_selected_metric_score = float(
                np.dot(self.last_metric_weights, self.last_selected_metric_components)
            )
        else:
            self.last_selected_metric_components = np.zeros(3, dtype=np.float32)
            self.last_selected_metric_score = 0.0

    def recompute_selected_aggregation_weights(self, selected_neighbors):
        if not selected_neighbors:
            self.last_aggregation_weights = {}
            self.last_selected_alpha = {}
            self.update_selected_metric_summary(selected_neighbors)
            return {}

        total_samples = sum(self.neighbor_sample_num_dict.get(n, 0) for n in selected_neighbors) + 1e-8
        raw_weights = {}
        for nei_id in selected_neighbors:
            sim = float(self.model_similarity_list.get(nei_id, 0.0))
            feat = float(self.feature_complementarity_list.get(nei_id, 0.0))
            logit = float(self.logits_complementarity_list.get(nei_id, 0.0))
            score = float(
                self.last_metric_weights[0] * sim
                + self.last_metric_weights[1] * feat
                + self.last_metric_weights[2] * logit
            )
            sample_ratio = float(self.neighbor_sample_num_dict.get(nei_id, 0) / total_samples)
            raw_weights[nei_id] = max(score, 0.0) * max(sample_ratio, 1e-8)

        weight_sum = sum(raw_weights.values())
        if weight_sum <= 1e-12:
            uniform = 1.0 / len(selected_neighbors)
            aggregation_weights = {nei_id: uniform for nei_id in selected_neighbors}
        else:
            aggregation_weights = {nei_id: float(w / (weight_sum + 1e-8)) for nei_id, w in raw_weights.items()}

        self.last_aggregation_weights = aggregation_weights
        self.last_selected_alpha = aggregation_weights.copy()
        self.update_selected_metric_summary(selected_neighbors)
        return aggregation_weights
    # =========================================================
    # RL neighbor selection (learn metric fusion weights)
    # =========================================================
    def select_neighbors_rl(self, round_idx: int, explore: bool = True):
        state = self.build_state(round_idx)

        metric_weights, log_prob, value, policy_mean = self.rl_agent.select_action(state, explore)

        self.rl_states.append(np.array(state, copy=True))
        self.rl_actions.append(np.array(metric_weights, dtype=np.float32, copy=True))
        self.rl_log_probs.append(float(log_prob))
        self.rl_values.append(float(value))
        self.last_action_scores = np.array(policy_mean, copy=True)

        selected_neighbors, aggregation_weights, blended_weights = self._decode_weights(metric_weights, round_idx)
        self.last_metric_weights = np.array(blended_weights, copy=True)

        if round_idx % 10 == 0:
            self.logger.info(
                f"[Client {self.client_idx}] Round {round_idx} - Metric weights: "
                f"{[round(float(x), 4) for x in blended_weights]}"
            )
            self.logger.info(f"[Client {self.client_idx}] Round {round_idx} - Selected neighbors: {selected_neighbors}")
            self.logger.info(f"[Client {self.client_idx}] Aggregation weights: {aggregation_weights}")

        self.last_selected_neighbors = selected_neighbors
        self.last_aggregation_weights = aggregation_weights
        self.last_selected_alpha = aggregation_weights.copy()

        return selected_neighbors, aggregation_weights

    def _decode_weights(self, metric_weights: np.ndarray, round_idx: int = 0):
        metric_weights = np.asarray(metric_weights, dtype=np.float32).reshape(-1)
        if metric_weights.size != 3:
            metric_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        metric_weights = np.clip(metric_weights, 1e-8, None)
        metric_weights = metric_weights / (metric_weights.sum() + 1e-8)

        heuristic = np.array(
            [
                float(getattr(self.args, "heuristic_sim_weight", 0.2)),
                float(getattr(self.args, "heuristic_feat_weight", 0.4)),
                float(getattr(self.args, "heuristic_logit_weight", 0.4)),
            ],
            dtype=np.float32,
        )
        heuristic = heuristic / (heuristic.sum() + 1e-8)
        prior_blend = float(getattr(self.args, "ppo_weight_prior_blend", 0.5))
        blended_weights = prior_blend * heuristic + (1.0 - prior_blend) * metric_weights
        blended_weights = blended_weights / (blended_weights.sum() + 1e-8)

        neighbor_ids = [i for i in range(self.args.client_num_in_total) if i != self.client_idx]
        client_num_per_round = getattr(self.args, "client_num_per_round", 5)
        k = min(client_num_per_round, len(neighbor_ids))
        if k <= 0:
            return [], {}, blended_weights

        total_samples = sum(self.neighbor_sample_num_dict.get(n, 0) for n in neighbor_ids) + 1e-8
        score_rows = []
        progress = float(round_idx / max(1, getattr(self.args, "comm_round", 1) - 1))
        stale_decay = float(getattr(self.args, "staleness_decay", 0.05))
        refresh_bonus = float(getattr(self.args, "refresh_bonus", 0.02)) * (1.0 - progress)
        stale_cap = max(1.0, float(getattr(self.args, "staleness_cap", 20.0)))
        for nei_id in neighbor_ids:
            sim = float(self.model_similarity_list.get(nei_id, 0.0))
            feat = float(self.feature_complementarity_list.get(nei_id, 0.0))
            logit = float(self.logits_complementarity_list.get(nei_id, 0.0))
            staleness = float(self.neighbor_staleness_dict.get(nei_id, 0))
            observed = float(self.neighbor_observed_dict.get(nei_id, 0.0))
            base_score = float(blended_weights[0] * sim + blended_weights[1] * feat + blended_weights[2] * logit)
            stale_factor = float(np.exp(-stale_decay * min(staleness, stale_cap)))
            info_bonus = refresh_bonus * min(staleness, stale_cap) / stale_cap
            score = base_score * stale_factor + info_bonus
            sample_ratio = float(self.neighbor_sample_num_dict.get(nei_id, 0) / total_samples)
            score_rows.append((nei_id, score, sample_ratio, sim, feat, logit, staleness, observed))

        sample_mode = str(getattr(self.args, "neighbor_sample_mode", "prob")).strip().lower()
        if sample_mode in {"topk", "deterministic"}:
            selected_rows = sorted(score_rows, key=lambda item: item[1], reverse=True)[:k]
        else:
            scores = np.asarray([row[1] for row in score_rows], dtype=np.float64)
            tau_start = float(getattr(self.args, "neighbor_sample_tau", 0.3))
            tau_min = float(getattr(self.args, "neighbor_sample_tau_min", 0.05))
            sample_tau = max(1e-6, tau_min + (tau_start - tau_min) * (1.0 - progress))
            logits = (scores - np.max(scores)) / sample_tau
            probs = np.exp(logits)
            prob_sum = float(probs.sum())
            if (not np.isfinite(prob_sum)) or prob_sum <= 1e-12:
                probs = np.ones(len(score_rows), dtype=np.float64) / max(len(score_rows), 1)
            else:
                probs = probs / prob_sum
            selected_indices = np.random.choice(
                len(score_rows),
                size=k,
                replace=False,
                p=probs,
            )
            selected_rows = [score_rows[int(idx)] for idx in selected_indices]

        positive_mass = sum(max(row[1], 0.0) * max(row[2], 1e-8) for row in selected_rows)
        if positive_mass <= 1e-12:
            fallback_neighbors = np.random.choice(neighbor_ids, size=k, replace=False).tolist()
            uniform_weight = 1.0 / len(fallback_neighbors)
            self.last_selected_metric_score = 0.0
            self.last_selected_metric_components = np.zeros(3, dtype=np.float32)
            return fallback_neighbors, {nei_id: uniform_weight for nei_id in fallback_neighbors}, blended_weights

        aggregation_weights = {}
        comp_acc = np.zeros(3, dtype=np.float32)
        score_acc = 0.0
        for nei_id, score, sample_ratio, sim, feat, logit, _, _ in selected_rows:
            raw_weight = max(score, 0.0) * max(sample_ratio, 1e-8)
            aggregation_weights[nei_id] = raw_weight
            comp_acc += np.array([sim, feat, logit], dtype=np.float32)
            score_acc += score

        weight_sum = sum(aggregation_weights.values()) + 1e-8
        aggregation_weights = {nei_id: float(w / weight_sum) for nei_id, w in aggregation_weights.items()}
        self.last_selected_metric_components = comp_acc / max(len(selected_rows), 1)
        self.last_selected_metric_score = float(score_acc / max(len(selected_rows), 1))
        return [row[0] for row in selected_rows], aggregation_weights, blended_weights

    def reset_rl_buffers(self):
        """Reset per-step RL buffers."""
        self.rl_states = []
        self.rl_actions = []
        self.rl_log_probs = []
        self.rl_rewards = []
        self.rl_values = []

    def _reset_proto_stats(self):
        """Reset per-class prototype statistics before local training."""
        num_classes = self.args.num_classes
        self.proto_class_feat_sums = [None for _ in range(num_classes)]
        self.proto_class_counts = [0 for _ in range(num_classes)]
        self.proto_global_feat_sum = None
        self.proto_global_count = 0

    def _update_proto_stats(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Update prototype statistics with a batch of features/labels.
        feats: [B, feat_dim], labels: [B]
        """
        num_classes = self.args.num_classes

        if self.proto_class_feat_sums is None or self.proto_class_counts is None:
            self._reset_proto_stats()

        if self.proto_global_feat_sum is None:
            self.proto_global_feat_sum = feats.detach().sum(dim=0)
        else:
            self.proto_global_feat_sum += feats.detach().sum(dim=0)
        self.proto_global_count += feats.size(0)

        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                selected = feats[mask]
                if self.proto_class_feat_sums[c] is None:
                    self.proto_class_feat_sums[c] = selected.detach().sum(dim=0)
                else:
                    self.proto_class_feat_sums[c] += selected.detach().sum(dim=0)
                self.proto_class_counts[c] += int(mask.sum().item())

    def _compute_selection_diversity(self) -> float:
        """Entropy of selected neighbor weights, normalized to [0,1]."""
        if not self.last_selected_alpha:
            return 0.0

        weights = np.array(list(self.last_selected_alpha.values()), dtype=np.float32)
        s = float(weights.sum())
        if s <= 0:
            return 0.0
        weights = weights / s
        entropy = -float(np.sum(weights * np.log(weights + 1e-8)))
        max_entropy = np.log(len(weights) + 1e-8)
        if max_entropy <= 0:
            return 0.0
        return float(entropy / max_entropy)

    def _update_behavior_stats(self, acc_after: float):
        try:
            loss_after = float(self.model_trainer.eval_loss(self.local_test_data, self.device))
        except Exception:
            loss_after = float(self.last_val_loss)

        acc_delta = float(acc_after) - float(self.last_val_acc)
        loss_delta = float(self.last_val_loss) - float(loss_after)

        entropy_mean = 0.0
        maxprob_mean = 0.0
        margin_mean = 0.0
        try:
            batch = next(iter(self.local_test_data))
            x, _ = batch
            x = x.to(self.device)
            with torch.no_grad():
                logits = self.model_trainer.model.to(self.device)(x)
                probs = torch.softmax(logits, dim=1)
                maxprob, _ = probs.max(dim=1)
                top2 = torch.topk(probs, k=2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropy_mean = float(entropy.mean().item())
                maxprob_mean = float(maxprob.mean().item())
                margin_mean = float(margin.mean().item())
        except Exception:
            pass

        self.behavior_stats = np.array(
            [entropy_mean, maxprob_mean, margin_mean, acc_delta, loss_delta],
            dtype=np.float32,
        )
        self.last_val_loss = float(loss_after)
        self.last_val_acc = float(acc_after)

    def normalize_reward(self, reward: float) -> float:
        beta = float(getattr(self.args, "reward_norm_beta", 0.05))
        reward = float(reward)
        delta = reward - self.diag_reward_ema
        self.diag_reward_ema += beta * delta
        self.diag_reward_var_ema = (1.0 - beta) * self.diag_reward_var_ema + beta * (delta ** 2)
        reward_std = float(np.sqrt(max(self.diag_reward_var_ema, 1e-8)))
        normalized_reward = (reward - self.diag_reward_ema) / reward_std
        reward_clip = float(getattr(self.args, "reward_clip", 2.0))
        return float(np.clip(normalized_reward, -reward_clip, reward_clip))

    def build_state(self, round_idx: int) -> np.ndarray:
        """Build a fixed-size summary state for metric-weight PPO."""
        self_loss = float(self.expected_val_loss) if self.expected_val_loss is not None else 1.0
        self_acc = float(self.expected_acc) if self.expected_acc is not None else 0.0
        self_state = np.array(
            [
                self_loss,
                self_acc,
                float(self.delta_theta_norm),
                float(self.behavior_stats[4]),
            ],
            dtype=np.float32,
        )

        stale_cap = max(1.0, float(getattr(self.args, "staleness_cap", 20.0)))
        sim_values = []
        feat_values = []
        logit_values = []
        stale_values = []
        observed_values = []

        for other_id in range(self.args.client_num_in_total):
            if other_id == self.client_idx:
                continue
            sim_values.append(float(self.model_similarity_list.get(other_id, 0.0)))
            feat_values.append(float(self.feature_complementarity_list.get(other_id, 0.0)))
            logit_values.append(float(self.logits_complementarity_list.get(other_id, 0.0)))
            stale_values.append(
                min(float(self.neighbor_staleness_dict.get(other_id, 0)), stale_cap) / stale_cap
            )
            observed_values.append(float(self.neighbor_observed_dict.get(other_id, 0.0)))

        def _summary(values):
            if not values:
                return [0.0, 0.0, 0.0]
            arr = np.asarray(values, dtype=np.float32)
            return [float(arr.mean()), float(arr.std()), float(arr.max())]

        neighbor_summary = np.array(
            _summary(sim_values) + _summary(feat_values) + _summary(logit_values),
            dtype=np.float32,
        )

        stale_mean = float(np.mean(stale_values)) if stale_values else 0.0
        stale_max = float(np.max(stale_values)) if stale_values else 0.0
        observed_ratio = float(np.mean(observed_values)) if observed_values else 0.0
        staleness_summary = np.array([stale_mean, stale_max, observed_ratio], dtype=np.float32)

        policy_summary = np.array(
            [
                float(self.last_metric_weights[0]),
                float(self.last_metric_weights[1]),
                float(self.last_metric_weights[2]),
                float(self.last_selected_metric_components[0]),
                float(self.last_selected_metric_components[1]),
                float(self.last_selected_metric_components[2]),
            ],
            dtype=np.float32,
        )

        full_state = np.concatenate([self_state, neighbor_summary, staleness_summary, policy_summary], axis=0)
        return full_state

    # =========================================================
    # RL 更新接口（当前仍然是“单步 PPO”，如需滑动窗口只改此函数逻辑）
    # =========================================================
    def update_rl_agent(self, reward: float, round_idx: int = None):
        self.reward_history.append(float(reward))

        if self.rl_agent is None:
            return None

        self.rl_rewards.append(float(reward))

        metrics = None
        rollout_len = getattr(self.args, "rl_rollout_len", 5)

        lens = [
            len(self.rl_states),
            len(self.rl_actions),
            len(self.rl_log_probs),
            len(self.rl_rewards),
            len(self.rl_values),
        ]
        T_min = min(lens)
        T_max = max(lens)

        if T_min == T_max and T_min >= rollout_len:
            states_window = self.rl_states[-rollout_len:]
            actions_window = self.rl_actions[-rollout_len:]
            log_probs_window = self.rl_log_probs[-rollout_len:]
            rewards_window = self.rl_rewards[-rollout_len:]
            values_window = self.rl_values[-rollout_len:]

            metrics = self.rl_agent.update(
                states=states_window,
                actions=actions_window,
                log_probs=log_probs_window,
                rewards=rewards_window,
                values=values_window,
            )

            # keep sliding window
            self.rl_states = self.rl_states[-rollout_len:]
            self.rl_actions = self.rl_actions[-rollout_len:]
            self.rl_log_probs = self.rl_log_probs[-rollout_len:]
            self.rl_rewards = self.rl_rewards[-rollout_len:]
            self.rl_values = self.rl_values[-rollout_len:]
        elif T_min != T_max:
            self.logger.warning(
                f"[Client {self.client_idx}] RL buffers mismatch, drop: "
                f"states={lens[0]}, actions={lens[1]}, log_probs={lens[2]}, "
                f"rewards={lens[3]}, values={lens[4]}"
            )
            self.reset_rl_buffers()

        return metrics

    def _try_update_rl_trajectory(self):
        """
        当 5 个缓存列表长度一致时，执行一次 PPO 更新，并清空缓存。
        如果不一致，则直接丢弃该轨迹（避免 reward 与 value/state 错位）。
        """
        if self.rl_agent is None:
            return None

        lens = [
            len(self.rl_states),
            len(self.rl_actions),
            len(self.rl_log_probs),
            len(self.rl_rewards),
            len(self.rl_values),
        ]
        T_min = min(lens)
        T_max = max(lens)

        if T_min == 0:
            return None

        if T_min == T_max:
            loss = self.rl_agent.update(
                states=self.rl_states,
                actions=self.rl_actions,
                log_probs=self.rl_log_probs,
                rewards=self.rl_rewards,
                values=self.rl_values,
            )
            self.reset_rl_buffers()
            return loss
        else:
            self.logger.warning(
                f"[Client {self.client_idx}] RL buffers mismatch, drop: "
                f"states={lens[0]}, actions={lens[1]}, log_probs={lens[2]}, "
                f"rewards={lens[3]}, values={lens[4]}"
            )
            self.reset_rl_buffers()
            return None

        def update_rl_agent(self, reward: float, round_idx: int = None):
            self.reward_history.append(float(reward))

        if self.rl_agent is None:
            return None

        self.rl_rewards.append(float(reward))

        metrics = None

        rollout_len = getattr(self.args, "rl_rollout_len", 5)
        lens = [
            len(self.rl_states),
            len(self.rl_actions),
            len(self.rl_log_probs),
            len(self.rl_rewards),
            len(self.rl_values),
        ]
        T_min = min(lens)
        T_max = max(lens)

        if T_min == T_max and T_min >= rollout_len:
            states_window = self.rl_states[-rollout_len:]
            actions_window = self.rl_actions[-rollout_len:]
            log_probs_window = self.rl_log_probs[-rollout_len:]
            rewards_window = self.rl_rewards[-rollout_len:]
            values_window = self.rl_values[-rollout_len:]

            metrics = self.rl_agent.update(
                states=states_window,
                actions=actions_window,
                log_probs=log_probs_window,
                rewards=rewards_window,
                values=values_window,
            )

            # keep sliding window
            self.rl_states = self.rl_states[-rollout_len:]
            self.rl_actions = self.rl_actions[-rollout_len:]
            self.rl_log_probs = self.rl_log_probs[-rollout_len:]
            self.rl_rewards = self.rl_rewards[-rollout_len:]
            self.rl_values = self.rl_values[-rollout_len:]
        elif T_min != T_max:
            self.logger.warning(
                f"[Client {self.client_idx}] RL buffers mismatch, drop: "
                f"states={lens[0]}, actions={lens[1]}, log_probs={lens[2]}, "
                f"rewards={lens[3]}, values={lens[4]}"
            )
            self.reset_rl_buffers()

        if getattr(self.args, "enable_diagnostics", True):
            interval = getattr(self.args, "diag_log_interval", 1)
            if round_idx is None:
                do_log = True
            else:
                do_log = (round_idx % interval == 0)

            if do_log:
                if isinstance(metrics, dict):
                    self.logger.info(
                        f"[Diag][Client {self.client_idx}] PPO: "
                        f"clip={metrics.get('ppo/clip_frac', 0):.3f}, "
                        f"ratio={metrics.get('ppo/ratio_mean', 0):.3f}?{metrics.get('ppo/ratio_std', 0):.3f}, "
                        f"adv={metrics.get('ppo/adv_mean', 0):.3f}?{metrics.get('ppo/adv_std', 0):.3f}, "
                        f"ret={metrics.get('ppo/ret_mean', 0):.3f}?{metrics.get('ppo/ret_std', 0):.3f}, "
                        f"V={metrics.get('ppo/v_mean', 0):.3f}?{metrics.get('ppo/v_std', 0):.3f}, "
                        f"loss(pi)={metrics.get('ppo/policy_loss', 0):.6f}, "
                        f"loss(v)={metrics.get('ppo/value_loss', 0):.6f}, "
                        f"H={metrics.get('ppo/entropy', 0):.3f}"
                    )

        return metrics

    # def update_rl_agent(self, reward: float):
    #     """
    #     使用奖励更新 RL 代理（滑动窗口形式）：
    #     - 缓存所有历史的 state / action / log_prob / value / reward
    #     - 每次 reward 到来时，若历史长度 >= rl_rollout_len，
    #       则取最近 rl_rollout_len 步构成一个窗口，做一次 PPO 更新
    #     - 不清空缓存，形成滑动窗口（window_t, window_{t+1}, ...）
    #     """
    #     # 统计用：记录 reward 历史（不参与 PPO）
    #     self.reward_history.append(float(reward))
    #
    #     if self.rl_agent is None:
    #         return None
    #
    #     # 追加当前轮的 reward 到缓存
    #     self.rl_rewards.append(float(reward))
    #
    #     # --- 计算当前各缓存长度 ---
    #     lens = [
    #         len(self.rl_states),
    #         len(self.rl_actions),
    #         len(self.rl_log_probs),
    #         len(self.rl_rewards),
    #         len(self.rl_values),
    #     ]
    #     T_min = min(lens)
    #     T_max = max(lens)
    #
    #     # 如果有列表还没对齐，先不更新（避免错位）
    #     if T_min == 0 or T_min != T_max:
    #         # 可选：打开调试
    #         # self.logger.warning(
    #         #     f"[Client {self.client_idx}] RL buffers mismatch, skip update: "
    #         #     f"states={lens[0]}, actions={lens[1]}, log_probs={lens[2]}, "
    #         #     f"rewards={lens[3]}, values={lens[4]}"
    #         # )
    #         return None
    #
    #     # 此时说明所有列表长度都为 T_min
    #     T = T_min
    #
    #     # --- 如果历史太短，不足一个 rollout_len，只缓存不更新 ---
    #     rollout_len = getattr(self.args, "rl_rollout_len", 5)  # 超参，可在 args 中设置
    #     if T < rollout_len:
    #         return None
    #
    #     # --- 使用最近窗口：取最近 rollout_len 步组成一个 trajectory ---
    #     T_use = rollout_len
    #     states_window = self.rl_states[-T_use:]
    #     actions_window = self.rl_actions[-T_use:]
    #     log_probs_window = self.rl_log_probs[-T_use:]
    #     rewards_window = self.rl_rewards[-T_use:]
    #     values_window = self.rl_values[-T_use:]
    #
    #     # 调用 PPO 更新（内部假设五个列表长度相等）
    #     loss = self.rl_agent.update(
    #         states=states_window,
    #         actions=actions_window,
    #         log_probs=log_probs_window,
    #         rewards=rewards_window,
    #         values=values_window,
    #     )
    #
    #     # 注意：这里**没有清空缓存**，历史继续累积，形成滑动窗口
    #     return loss

    def _log_policy_diagnostics(self, round_idx, selected_neighbors, aggregation_weights):
        # 1) 权重向量统计
        if aggregation_weights:
            w = np.array(list(aggregation_weights.values()), dtype=np.float32)
            w = w / (w.sum() + 1e-8)
            entropy = -float(np.sum(w * np.log(w + 1e-8)))
            max_w = float(np.max(w))
            eff_n = float(1.0 / (np.sum(w ** 2) + 1e-8))  # effective number of neighbors
        else:
            entropy, max_w, eff_n = 0.0, 0.0, 0.0

        # 2) action_scores 统计
        if self.last_action_scores is not None and self.last_action_scores.size > 0:
            s_mean = float(np.mean(self.last_action_scores))
            s_std = float(np.std(self.last_action_scores))
            s_min = float(np.min(self.last_action_scores))
            s_max = float(np.max(self.last_action_scores))
        else:
            s_mean = s_std = s_min = s_max = 0.0

        # 3) 被选邻居的相似/互补性统计（看策略是否真的在用这些特征）
        sims, cfeat, clogits = [], [], []
        for j in selected_neighbors:
            sims.append(self.model_similarity_list.get(j, 0.0))
            cfeat.append(self.feature_complementarity_list.get(j, 0.0))
            clogits.append(self.logits_complementarity_list.get(j, 0.0))

        def _mean_std(x):
            if len(x) == 0:
                return 0.0, 0.0
            x = np.array(x, dtype=np.float32)
            return float(x.mean()), float(x.std())

        sim_m, sim_s = _mean_std(sims)
        cf_m, cf_s = _mean_std(cfeat)
        cl_m, cl_s = _mean_std(clogits)

        self.logger.info(
            f"[Diag][Client {self.client_idx}][Round {round_idx}] "
            f"sel={selected_neighbors} | "
            f"W: ent={entropy:.3f}, max={max_w:.3f}, effN={eff_n:.2f} | "
            f"scores: mean={s_mean:.3f}, std={s_std:.3f}, min={s_min:.3f}, max={s_max:.3f} | "
            f"sel-metrics: sim={sim_m:.3f}±{sim_s:.3f}, "
            f"cfeat={cf_m:.3f}±{cf_s:.3f}, clogits={cl_m:.3f}±{cl_s:.3f}"
        )

    # =========================================================
    # 邻居选择：warmup 随机选择
    # =========================================================
    def get_random_neighbors(self):
        """随机选择邻居（用于 warmup 阶段）"""
        all_neighbors = [i for i in range(self.args.client_num_in_total) if i != self.client_idx]
        client_num_per_round = getattr(self.args, "client_num_per_round", 5)

        if len(all_neighbors) == 0:
            return [], {}

        selected_neighbors = np.random.choice(
            all_neighbors,
            size=min(client_num_per_round, len(all_neighbors)),
            replace=False,
        ).tolist()

        aggregation_weights = {}
        uniform_weight = 1.0 / len(selected_neighbors)
        for nei_id in selected_neighbors:
            aggregation_weights[nei_id] = uniform_weight

        self.last_selected_neighbors = selected_neighbors
        self.last_aggregation_weights = aggregation_weights
        self.last_selected_alpha = aggregation_weights.copy()

        return selected_neighbors, aggregation_weights

    # =========================================================
    # 其他工具方法：随机特征 / 通信评分 / 数据集更新 / 特征提取
    # =========================================================
    def generate_random_features(self) -> torch.Tensor:
        """生成用于 soft logits 的随机特征，范围 [0,1)"""
        random_features = torch.rand(self.batch_size, self.input_size).to(self.device)
        return random_features

    def update_communication_score(self, other_client_idx, score):
        self.communication_scores[other_client_idx] = score

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self) -> int:
        return self.local_sample_number

    def extract_features_and_compute_soft_logits(self):
        """
        返回：
            extract_features: PCA子空间基 self.pca_basis（用于特征互补性）
            soft_logits_mean: 使用“缓存的类别原型 + 频率平滑”得到的 soft logits 向量
        """
        model = self.model_trainer.model.to(self.device)
        model.eval()

        # ==========================
        # 1) PCA 子空间（仅首次拟合）
        # ==========================
        if not self.pca_fitted:
            features = []
            with torch.no_grad():
                for x, labels in self.local_training_data:
                    x = x.to(self.device)
                    feat = model.base(x)
                    features.append(feat.cpu())

            if len(features) == 0:
                feat_dim = getattr(self.args, "feature_dim", self.input_size)
                n_components = getattr(self.args, "num_components", min(16, feat_dim))
                self.pca_basis = torch.eye(
                    n_components,
                    feat_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                features = torch.cat(features, dim=0)
                feat_np = features.numpy()
                n_components = getattr(self.args, "num_components", 16)
                pca = PCA(n_components=n_components)
                pca.fit(feat_np)
                self.pca_basis = torch.tensor(
                    pca.components_,
                    device=self.device,
                    dtype=torch.float32,
                )
            self.pca_fitted = True

        extract_features = self.pca_basis

        # ==========================
        # 2) 原型 + 频率平滑 soft logits（基于缓存统计）
        # ==========================
        num_classes = self.args.num_classes

        if (
            self.proto_global_count == 0
            or self.proto_class_feat_sums is None
            or self.proto_class_counts is None
        ):
            soft_logits_mean = torch.full(
                (num_classes,),
                1.0 / num_classes,
                device=self.device,
                dtype=torch.float32,
            )
            return extract_features, soft_logits_mean

        global_mean_feat = self.proto_global_feat_sum / self.proto_global_count  # [feat_dim]

        prototypes = []
        proto_noise_std = getattr(self.args, "proto_noise_std", 0.01)
        for c in range(num_classes):
            if self.proto_class_counts[c] > 0 and self.proto_class_feat_sums[c] is not None:
                mu_c = self.proto_class_feat_sums[c] / self.proto_class_counts[c]
            else:
                noise = proto_noise_std * torch.randn_like(global_mean_feat)
                mu_c = global_mean_feat + noise
            prototypes.append(mu_c.unsqueeze(0))

        prototypes = torch.cat(prototypes, dim=0)  # [C, feat_dim]

        with torch.no_grad():
            logits = model.classifier(prototypes)  # [C, num_classes]
            temperature = getattr(self.args, "temperature", 1.0)
            per_class_soft = torch.softmax(logits / temperature, dim=1)  # [C, num_classes]

        alpha = getattr(self.args, "proto_dirichlet_alpha", 0.5)
        counts_tensor = torch.tensor(
            self.proto_class_counts,
            dtype=torch.float32,
            device=self.device,
        )  # [C]
        total_count = counts_tensor.sum()
        freq_smooth = (counts_tensor + alpha) / (total_count + alpha * num_classes + 1e-8)

        soft_logits_mean = torch.matmul(freq_smooth, per_class_soft)  # [num_classes]

        self.cached_soft_logits = soft_logits_mean.detach()

        return extract_features, soft_logits_mean

    # =========================================================
    # 本地训练与测试（训练中维护 delta_theta_norm / param_norm）
    # =========================================================
    def train(self, w_global, round_idx: int):
        """
        本地训练：
        - 加载聚合参数 w_global
        - 在本地数据上训练（在训练循环中通过 feature_hook 顺带统计原型）
        - 训练前后对参数展平，维护：
            self.delta_theta_norm = ||θ_after - θ_before|| / (||θ_before|| + eps)
            self.param_norm = ||θ_after||_2
        - 在本地测试集上评估 acc_after
        """
        # 训练前参数（聚合后的 θ_i^t）
        flat_before = _flatten_params(w_global)
        prev_norm = float(flat_before.norm().item()) if flat_before.numel() > 0 else 0.0

        # 加载聚合模型
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_id(self.client_idx)

        # 重置原型统计
        self._reset_proto_stats()

        def feature_hook(feats: torch.Tensor, labels: torch.Tensor):
            self._update_proto_stats(feats, labels)

        # 本地训练
        self.model_trainer.train(
            self.local_training_data,
            self.device,
            self.args,
            round_idx,
            self.logger,
            feature_hook=feature_hook,
        )

        # 训练后测试
        acc_after = self.model_trainer.test_after_aggregated(
            self.local_test_data,
            self.device,
        )

        self.expected_acc = (1 - self.expected_ema_eta) * self.expected_acc + self.expected_ema_eta * acc_after
        self._update_behavior_stats(acc_after)


        # 训练后参数
        trained_model = self.model_trainer.get_model_params()
        flat_after = _flatten_params(trained_model)
        after_norm = float(flat_after.norm().item()) if flat_after.numel() > 0 else 0.0

        # 更新自状态特征
        if flat_before.numel() > 0:
            delta_norm = float((flat_after - flat_before).norm().item())
            self.delta_theta_norm = delta_norm / (prev_norm + 1e-8)
        else:
            self.delta_theta_norm = 0.0

        self.param_norm = after_norm
        self.prev_flat_params = flat_after  # 如需今后更复杂统计可以用

        mdl = self.model_trainer.model

        return mdl, trained_model, acc_after

    def local_test(self, w, b_use_test_dataset: bool = True, global_test_data=None):
        """
        在本地或全局测试数据上评估
        """
        if global_test_data is not None:
            test_data = global_test_data
            data_distribution = self.data_distribution
        elif b_use_test_dataset:
            test_data = self.local_test_data
            data_distribution = None
        else:
            test_data = self.local_training_data
            data_distribution = None

        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(
            test_data,
            self.device,
            self.args,
            data_distribution,
        )
        return metrics

    def eval_on_val_loss(self, model_params):
        """
        在本地验证集上计算损失，用于 reward：R_t = L_val(before) - L_val(after)
        """
        return self.model_trainer.eval_loss_for_params(
            model_params, self.local_test_data, self.device
        )
