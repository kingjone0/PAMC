import copy
import gc
import logging
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from pamc.utils.slogits_visualizer import visualize_slogits_simple
from pamc.utils.attack import manipulate_one_model
from pamc.algorithm.client import Client


class BACAPI(object):
    """
    基础的去中心化个性化联邦学习训练框架（不使用 RL 进行客户端选择）。
    RLEnhancedBACAPI 将在此基础上，引入基于 POMDP 的 RL 策略。
    """

    def __init__(self, dataset, device, args, model_trainer, logger: logging.Logger):
        self.logger = logger
        self.device = device
        self.args = args

        # dataset: [train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions]
        (train_dataloaders,
         val_dataloaders,
         test_loader,
         data_local_num_dict,
         data_distributions) = dataset

        self.test_global = test_loader
        self.val_global = None
        self.client_list = []
        self.train_data_local_num_dict = data_local_num_dict
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.model_trainer = model_trainer
        self.final_round_slogits = []

        # 初始化客户端
        self._setup_clients(
            data_local_num_dict,
            train_dataloaders,
            val_dataloaders,
            model_trainer,
            data_distributions,
            getattr(self.args, "expected_acc", 0.5),
        )

        # 统计信息
        self.init_stat_info()

    # ----------------------------------------------------------------------
    # 初始化与基础工具
    # ----------------------------------------------------------------------
    def _setup_clients(self,
                       train_data_local_num_dict,
                       train_data_local_dict,
                       test_data_local_dict,
                       model_trainer,
                       data_distributions,
                       expected_acc: float):
        """
        根据本地数据构造每个客户端的 Client 实例。
        """
        self.logger.info("############ setup_clients (START) #############")
        self.client_list = []
        for client_idx in range(self.args.client_num_in_total):
            c = Client(
                client_idx=client_idx,
                local_training_data=train_data_local_dict[client_idx],
                local_test_data=test_data_local_dict[client_idx],
                local_sample_number=train_data_local_num_dict[client_idx],
                args=self.args,
                device=self.device,
                model_trainer=model_trainer,
                data_distribution=data_distributions[client_idx],
                expected_acc=expected_acc,
                logger=self.logger,
            )
            self.client_list.append(c)
        for client in self.client_list:
            for other_idx in range(self.args.client_num_in_total):
                if other_idx != client.client_idx:
                    client.neighbor_sample_num_dict[other_idx] = train_data_local_num_dict[other_idx]
        self.logger.info("############ setup_clients (END) #############")

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_generalized_acc"] = []
        self.stat_info["global_personalized_acc"] = []
        self.stat_info["local_val_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["best_personalized_acc"] = -1.0
        self.stat_info["best_personalized_round"] = -1
        self.stat_info["best_generalized_acc_at_best_personalized"] = 0.0

    def _select_malicious_clients(self):
        total = self.args.client_num_in_total
        if getattr(self.args, 'byzantine_clients', ''):
            ids = [int(x) for x in self.args.byzantine_clients.split(',') if x.strip() != '']
        else:
            k = int(total * getattr(self.args, 'byzantine_ratio', 0.0))
            ids = list(np.random.choice(total, k, replace=False)) if k > 0 else []
        ids = [i for i in ids if 0 <= i < total]
        benign = [i for i in range(total) if i not in ids]
        return ids, benign

    # ----------------------------------------------------------------------
    # 训练主循环（基础版本：不使用 RL，作为对照）
    # ----------------------------------------------------------------------
    def train(self):
        """
        基础版本的训练流程：
        - 每轮：对每个客户端
          * 计算特征与 soft logits
          * 更新互补性图
          * 对邻居进行启发式加权聚合
          * 执行本地训练
        - 每轮结束：在所有客户端上评估性能
        """
        w_per_models = []
        w_per_mods = []
        for client in self.client_list:
            w_per_models.append(copy.deepcopy(client.model_trainer.get_model_params()))
            w_per_mods.append(copy.deepcopy(client.model_trainer.model))

        malicious_ids, benign_ids = self._select_malicious_clients()

        for round_idx in tqdm(range(self.args.comm_round)):
            self.logger.info("################ Communication round : {}".format(round_idx))
            print("communicate round : {}".format(round_idx))

            # 1) 提取特征子空间与 soft logits
            w_per_feats = []
            w_per_slogits = []

            w_per_models_lstrd = copy.deepcopy(w_per_models)
            w_per_mods_lstrd = copy.deepcopy(w_per_mods)

            if malicious_ids:
                for mid in malicious_ids:
                    manipulate_one_model(self.args, w_per_mods_lstrd[mid], mid, global_model=w_per_mods_lstrd[mid])
                    w_per_models_lstrd[mid] = copy.deepcopy(w_per_mods_lstrd[mid].state_dict())

            for client in self.client_list:
                client.model_trainer.set_model_params(w_per_models_lstrd[client.client_idx])
                client.model_trainer.model.to(self.device)
                client.model_trainer.model.eval()

                client_features, client_slogits = client.extract_features_and_compute_soft_logits()
                w_per_feats.append(client_features)
                w_per_slogits.append(client_slogits)

            # 2) 启发式邻居聚合 + 本地训练
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info("@@@@@@@@@@@@@@@@ Training Client ({}, {})".format(round_idx, clnt_idx))

                client = self.client_list[clnt_idx]

                nei_indexs = [i for i in range(self.args.client_num_in_total) if i != clnt_idx]

                self.update_communication_graph(
                    client,
                    nei_indexs,
                    w_per_models_lstrd,
                    w_per_feats,
                    w_per_slogits,
                )

                if clnt_idx not in nei_indexs:
                    nei_indexs.append(clnt_idx)
                nei_indexs = sorted(nei_indexs)

                aggregated_params = self._aggregate_whole_model(
                    round_idx, clnt_idx, nei_indexs, w_per_models_lstrd
                )
                loss_after_agg = client.eval_on_val_loss(aggregated_params)
                expected_acc_before = float(client.expected_acc)
                loss_after_agg = client.eval_on_val_loss(aggregated_params)
                expected_acc_before = float(client.expected_acc)

                trained_mdl, trained_model, acc_after = client.train(
                    aggregated_params, round_idx
                )

                w_per_models[clnt_idx] = copy.deepcopy(trained_model)
                w_per_mods[clnt_idx] = copy.deepcopy(trained_mdl)

                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if getattr(self.args, "eval_benign_only", False):
                self._test_on_all_clients(w_per_models, round_idx, client_indices=benign_ids)
            else:
                self._test_on_all_clients(w_per_models, round_idx)

        self.final_round_slogits = []
        for client in self.client_list:
            _, client_slogits = client.extract_features_and_compute_soft_logits()
            self.final_round_slogits.append(copy.deepcopy(client_slogits))
        self.visualize_slogits()
        self._log_final_personalized_acc_summary()

    # ----------------------------------------------------------------------
    # 模型聚合（整模型）
    # ----------------------------------------------------------------------
    def _aggregate_whole_model(self, round_idx, cur_clnt, nei_indexs, w_per_models):
        """
        聚合整个模型，不区分特征提取层和分类器。
        """
        self.logger.info(f"Aggregating whole model for client {cur_clnt}")

        client = self.client_list[cur_clnt]

        tau = getattr(self.args, "trust_region_tau", 0.8)
        alpha_min = getattr(self.args, "alpha_min", 0.1)

        sample_numbers = [self.client_list[nei_idx].get_sample_number() for nei_idx in nei_indexs]
        total_samples = sum(sample_numbers)
        sample_ratios = [
            s / total_samples if total_samples > 0 else 1.0 / max(len(nei_indexs), 1)
            for s in sample_numbers
        ]

        normalized_weights = None
        use_policy = (
            round_idx >= getattr(self.args, "warmup_rounds", 0)
            and len(getattr(client, "last_selected_alpha", {})) > 0
        )

        if use_policy:
            alpha_dict = {k: float(v) for k, v in client.last_selected_alpha.items()}
            weights_map = {nei: alpha_dict.get(nei, 0.0) for nei in nei_indexs if nei != cur_clnt}
            self_weight = max(alpha_min, 1.0 - sum(weights_map.values()))
            weights_map[cur_clnt] = self_weight
            w_list = [weights_map.get(nei, 0.0) for nei in nei_indexs]
            s = sum(w_list)
            if s > 1e-12:
                normalized_weights = [wi / s for wi in w_list]

        if (not use_policy) or (normalized_weights is None):
            w_sim = float(getattr(self.args, "heuristic_sim_weight", 0.2))
            w_feat = float(getattr(self.args, "heuristic_feat_weight", 0.4))
            w_logit = float(getattr(self.args, "heuristic_logit_weight", 0.4))
            similarities = [client.model_similarity_list.get(nei_idx, 0.0) for nei_idx in nei_indexs]
            feat_complementarities = [client.feature_complementarity_list.get(nei_idx, 0.0) for nei_idx in nei_indexs]
            logit_complementarities = [client.logits_complementarity_list.get(nei_idx, 0.0) for nei_idx in nei_indexs]
            combined_weights = [
                max(0.0, w_sim * sim + w_feat * feat_comp + w_logit * logit_comp) * ratio
                for sim, feat_comp, logit_comp, ratio in zip(
                    similarities, feat_complementarities, logit_complementarities, sample_ratios
                )
            ]
            total_weight = sum(combined_weights)
            if total_weight == 0:
                self.logger.warning("Total weight is zero, using sample ratios as fallback")
                normalized_weights = sample_ratios
            else:
                normalized_weights = [w / total_weight for w in combined_weights]

        aggregated_params = {}
        for param_name in w_per_models[cur_clnt].keys():
            aggregated_params[param_name] = torch.zeros_like(w_per_models[cur_clnt][param_name])
            for weight, nei_idx in zip(normalized_weights, nei_indexs):
                nei_param = w_per_models[nei_idx][param_name]
                aggregated_params[param_name] += weight * nei_param

        try:
            self_params = w_per_models[cur_clnt]
            for k in aggregated_params.keys():
                aggregated_params[k] = tau * aggregated_params[k] + (1.0 - tau) * self_params[k]
        except Exception as e:
            self.logger.warning(f"Trust-region mixing failed: {e}")

        return aggregated_params

    # ----------------------------------------------------------------------
    # 通信图 / 相似性与互补性计算
    # ----------------------------------------------------------------------
    def _extract_client_observation(self, client_idx, w_per_mdls):
        client = self.client_list[client_idx]
        client.model_trainer.set_model_params(w_per_mdls[client_idx])
        client.model_trainer.model.to(self.device)
        client.model_trainer.model.eval()
        return client.extract_features_and_compute_soft_logits()

    def update_communication_graph(self, client, nei_indexs, w_per_mdls, w_per_feats=None, w_per_slogits=None,
                                   round_idx=None):
        """
        更新相似性与互补性指标
        """
        device = self.device

        model_params_1 = w_per_mdls[client.client_idx]
        if w_per_feats is None or w_per_slogits is None:
            subspace_1, soft_logits_1 = self._extract_client_observation(client.client_idx, w_per_mdls)
            neighbor_subspace_list = []
            neighbor_soft_logits_list = []
            for other_client_idx in nei_indexs:
                subspace_j, soft_logits_j = self._extract_client_observation(other_client_idx, w_per_mdls)
                neighbor_subspace_list.append(subspace_j)
                neighbor_soft_logits_list.append(soft_logits_j)
        else:
            subspace_1 = w_per_feats[client.client_idx]
            soft_logits_1 = w_per_slogits[client.client_idx]
            neighbor_subspace_list = [w_per_feats[other_client_idx] for other_client_idx in nei_indexs]
            neighbor_soft_logits_list = [w_per_slogits[other_client_idx] for other_client_idx in nei_indexs]

        neighbor_model_params_list = [w_per_mdls[other_client_idx] for other_client_idx in nei_indexs]

        model_similarities = self.model_trainer.compute_model_similarity_batch_wrapper(
            model_params_1, neighbor_model_params_list
        )

        data_complementarities = torch.tensor(
            self.model_trainer.compute_data_complementarity_batch_wrapper(subspace_1, neighbor_subspace_list),
            device=device,
        )

        logits_complementarities = self.model_trainer.compute_logits_complementarity_batch_wrapper(
            soft_logits_1, neighbor_soft_logits_list
        )

        for idx, other_client_idx in enumerate(nei_indexs):
            sim = model_similarities[idx].item()
            sim_pos = (sim + 1.0) / 2.0
            feat_comp = max(0.0, min(1.0, data_complementarities[idx].item()))
            logit_comp = max(0.0, min(1.0, logits_complementarities[idx].item()))
            if round_idx is None:
                client.model_similarity_list[other_client_idx] = sim_pos
                client.feature_complementarity_list[other_client_idx] = feat_comp
                client.logits_complementarity_list[other_client_idx] = logit_comp
                client.neighbor_sample_num_dict[other_client_idx] = self.client_list[other_client_idx].get_sample_number()
            else:
                client.update_neighbor_metric_cache(
                    other_client_idx=other_client_idx,
                    sim=sim_pos,
                    feat_comp=feat_comp,
                    logit_comp=logit_comp,
                    sample_num=self.client_list[other_client_idx].get_sample_number(),
                    round_idx=round_idx,
                )

    # ----------------------------------------------------------------------
    # 测试与可视化
    # ----------------------------------------------------------------------
    def _test_on_all_clients(self, w_per_mdls, round_idx, client_indices=None):
        """
        在所有客户端上测试
        """
        self.logger.info("################ Test on all clients: Round {}".format(round_idx))

        global_generalized_acc = []
        global_personalized_acc = []
        global_avg_correct_conf = []
        global_avg_incorrect_conf = []

        local_val_acc = []
        local_val_weights = []

        if client_indices is None:
            client_indices = list(range(self.args.client_num_in_total))

        for client_idx in client_indices:
            client = self.client_list[client_idx]

            global_metrics = client.local_test(
                w_per_mdls[client_idx],
                True,
                self.test_global,
            )
            global_generalized_acc.append(global_metrics["generalized_acc"])
            global_personalized_acc.append(global_metrics["personalized_acc"])
            global_avg_correct_conf.append(global_metrics["avg_correct_conf"])
            global_avg_incorrect_conf.append(global_metrics["avg_incorrect_conf"])

            local_metrics = client.local_test(
                w_per_mdls[client_idx],
                True,
                None,
            )
            local_val_acc.append(local_metrics["generalized_acc"])
            local_val_weights.append(client.get_sample_number())

        avg_global_generalized_acc = float(np.mean(global_generalized_acc))
        avg_global_personalized_acc = float(np.mean(global_personalized_acc))
        avg_global_correct_conf = float(np.mean(global_avg_correct_conf))
        avg_global_incorrect_conf = float(np.mean(global_avg_incorrect_conf))

        total_weights = sum(local_val_weights)
        if total_weights > 0:
            avg_local_val_acc = float(
                sum(acc * weight for acc, weight in zip(local_val_acc, local_val_weights)) / total_weights
            )
        else:
            avg_local_val_acc = 0.0

        stats = {
            "Round": round_idx,
            "Global Test": {
                "Avg Generalized Acc": f"{avg_global_generalized_acc:.4f}",
                "Avg Personalized Acc": f"{avg_global_personalized_acc:.4f}",
                "Avg Correct Conf": f"{avg_global_correct_conf:.4f}",
                "Avg Incorrect Conf": f"{avg_global_incorrect_conf:.4f}",
            },
            "Local Validation": {
                "Avg Accuracy": f"{avg_local_val_acc:.4f}",
            },
        }

        print("\n" + "=" * 70)
        print(f"Round {round_idx} Evaluation Results:")
        print("-" * 70)
        print("Global Test Set:")
        print(f"  Generalized Accuracy: {avg_global_generalized_acc:.4f}")
        print(f"  Personalized Accuracy: {avg_global_personalized_acc:.4f}")
        print(f"  Avg Correct Confidence: {avg_global_correct_conf:.4f}")
        print(f"  Avg Incorrect Confidence: {avg_global_incorrect_conf:.4f}")
        print("-" * 70)
        print("Local Validation Set:")
        print(f"  Accuracy: {avg_local_val_acc:.4f}")
        print("=" * 70 + "\n")

        self.stat_info["global_generalized_acc"].append(avg_global_generalized_acc)
        self.stat_info["global_personalized_acc"].append(avg_global_personalized_acc)
        self.stat_info["local_val_acc"].append(avg_local_val_acc)
        if avg_global_personalized_acc > self.stat_info.get("best_personalized_acc", -1.0):
            self.stat_info["best_personalized_acc"] = avg_global_personalized_acc
            self.stat_info["best_personalized_round"] = round_idx
            self.stat_info["best_generalized_acc_at_best_personalized"] = avg_global_generalized_acc

        self.logger.info(stats)

    def _log_final_personalized_acc_summary(self, window_size=5):
        personalized_acc_history = self.stat_info["global_personalized_acc"]
        if not personalized_acc_history:
            return

        actual_window = min(window_size, len(personalized_acc_history))
        last_personalized_acc = personalized_acc_history[-actual_window:]
        avg_last_personalized_acc = float(np.mean(last_personalized_acc))
        start_round = len(personalized_acc_history) - actual_window
        end_round = len(personalized_acc_history) - 1

        summary = {
            "Final Personalized Acc Summary": {
                "Window Size": actual_window,
                "Round Range": f"{start_round}-{end_round}",
                "Last Personalized Acc": [f"{acc:.4f}" for acc in last_personalized_acc],
                "Avg Personalized Acc": f"{avg_last_personalized_acc:.4f}",
                "Best Personalized Acc": f"{self.stat_info.get('best_personalized_acc', 0.0):.4f}",
                "Best Personalized Round": self.stat_info.get("best_personalized_round", -1),
                "Generalized Acc At Best Personalized": f"{self.stat_info.get('best_generalized_acc_at_best_personalized', 0.0):.4f}",
            }
        }

        print("\n" + "=" * 70)
        print("Final Personalized Accuracy Summary:")
        print(f"  Rounds: {start_round} - {end_round}")
        print(f"  Last {actual_window} Personalized Acc: {[f'{acc:.4f}' for acc in last_personalized_acc]}")
        print(f"  Average Personalized Acc: {avg_last_personalized_acc:.4f}")
        print(f"  Best Personalized Acc: {self.stat_info.get('best_personalized_acc', 0.0):.4f}")
        print(f"  Best Personalized Round: {self.stat_info.get('best_personalized_round', -1)}")
        print(
            "  Generalized Acc At Best Personalized: "
            f"{self.stat_info.get('best_generalized_acc_at_best_personalized', 0.0):.4f}"
        )
        print("=" * 70 + "\n")

        self.logger.info(summary)

    def visualize_slogits(self):
        """
        可视化最终轮 soft logits
        """
        if self.final_round_slogits is not None:
            saved_files = visualize_slogits_simple(
                self.final_round_slogits,
                round_idx=self.args.comm_round - 1,
                output_dir=f"./slogits_plots_round_{self.args.comm_round - 1}",
            )
            self.logger.info(f"生成 {len(saved_files)} 张软logits柱状图")
            for filepath in saved_files:
                self.logger.info(f"保存图片: {filepath}")


class RLEnhancedBACAPI(BACAPI):
    def _log_feature_reward_correlation(self, feat_rows, reward_rows, round_idx):
        if not feat_rows or not reward_rows:
            return

        try:
            import numpy as _np
        except Exception:
            return

        X = _np.asarray(feat_rows, dtype=_np.float32)
        y = _np.asarray(reward_rows, dtype=_np.float32)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            return

        if X.shape[0] < 5:
            return

        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        denom = _np.sqrt((_np.sum(Xc ** 2, axis=0) + 1e-8) * (_np.sum(yc ** 2) + 1e-8))
        corr = _np.sum(Xc * yc[:, None], axis=0) / denom

        self.logger.info(
            f"[Diag][Round {round_idx}] corr(sim,feat,logits)="
            f"{corr[0]:.4f},{corr[1]:.4f},{corr[2]:.4f}"
        )

    """
    RL 增强版训练框架：在每个客户端内部使用 POMDP 的 RL 策略进行
    邻居选择与聚合权重决策。
    """

    def __init__(self, dataset, device, args, model_trainer, logger):
        super().__init__(dataset, device, args, model_trainer, logger)

    def train(self):
        """
        RL 增强版训练流程：每轮通过 RL 策略选择邻居并进行权重聚合，
        本地训练后根据验证损失改变量计算奖励（reward），
        并更新 RL 策略。
        """
        w_per_models = []
        w_per_mods = []
        for client in self.client_list:
            w_per_models.append(copy.deepcopy(client.model_trainer.get_model_params()))
            w_per_mods.append(copy.deepcopy(client.model_trainer.model))

        warmup_rounds = getattr(self.args, "warmup_rounds", 0)

        malicious_ids, benign_ids = self._select_malicious_clients()

        for round_idx in tqdm(range(self.args.comm_round)):
            self.logger.info("################ [RL] Communication round : {}".format(round_idx))
            print("RL communicate round : {}".format(round_idx))

            corr_feat_rows = []
            corr_reward_rows = []

            # 如果刚从 warmup 切换到 RL，重置各客户端的 RL 轨迹缓存
            if round_idx == warmup_rounds:
                for client in self.client_list:
                    client.reset_rl_buffers()

            # 1) 提取特征子空间与 soft logits
            w_per_models_lstrd = copy.deepcopy(w_per_models)
            w_per_mods_lstrd = copy.deepcopy(w_per_mods)

            # 2) 对每个客户端执行“RL 驱动的聚合 + 本地训练”
            for clnt_idx in range(self.args.client_num_in_total):
                self.logger.info("@@@@@@@@@@@@@@@@ [RL] Training Client ({}, {})".format(round_idx, clnt_idx))

                client = self.client_list[clnt_idx]
                client.advance_neighbor_staleness(round_idx)

                # 聚合前验证损失（上一轮参数）
                loss_before = client.eval_on_val_loss(w_per_models_lstrd[clnt_idx])

                # 邻居选择：warmup 用随机，其余用 RL
                if round_idx < warmup_rounds:
                    selected_neighbors, aggregation_weights = client.get_random_neighbors()
                else:
                    selected_neighbors, aggregation_weights = client.select_neighbors_rl(round_idx, explore=True)

                if selected_neighbors:
                    self.update_communication_graph(
                        client,
                        selected_neighbors,
                        w_per_models_lstrd,
                        round_idx=round_idx,
                    )
                    aggregation_weights = client.recompute_selected_aggregation_weights(selected_neighbors)

                nei_indexs = list(selected_neighbors)
                if clnt_idx not in nei_indexs:
                    nei_indexs.append(clnt_idx)
                nei_indexs = sorted(nei_indexs)

                self.logger.info(f"[RL] client {clnt_idx} selected neighbors = {selected_neighbors}")

                # 聚合
                aggregated_params = self._aggregate_whole_model(
                    round_idx, clnt_idx, nei_indexs, w_per_models_lstrd
                )

                # 更新选择计数（统计用）
                loss_after_agg = client.eval_on_val_loss(aggregated_params)
                expected_acc_before = float(client.expected_acc)
                for nei_idx in selected_neighbors:
                    if nei_idx != clnt_idx:
                        client.selection_count[nei_idx] = client.selection_count.get(nei_idx, 0) + 1

                # 本地训练
                trained_mdl, trained_model, acc_after = client.train(
                    aggregated_params, round_idx
                )

                loss_after_local = client.eval_on_val_loss(trained_mdl.state_dict())
                relative_agg_gain = float(
                    (loss_before - loss_after_agg) / (abs(loss_before) + 1e-8)
                )
                metric_gain = float(client.last_selected_metric_score)
                progress = float(round_idx / max(1, self.args.comm_round - 1))
                metric_weight = float(getattr(self.args, "reward_metric_weight", 0.1)) * (1.0 - 0.5 * progress)
                reward = (
                    float(getattr(self.args, "reward_agg_weight", 1.0)) * relative_agg_gain
                    + metric_weight * metric_gain
                )
                reward = client.normalize_reward(reward)

                # collect feature/reward for correlation (mean over selected neighbors)
                if selected_neighbors:
                    sims = [client.model_similarity_list.get(n, 0.0) for n in selected_neighbors]
                    feats = [client.feature_complementarity_list.get(n, 0.0) for n in selected_neighbors]
                    logs = [client.logits_complementarity_list.get(n, 0.0) for n in selected_neighbors]
                    corr_feat_rows.append([
                        float(sum(sims) / len(sims)),
                        float(sum(feats) / len(feats)),
                        float(sum(logs) / len(logs)),
                    ])
                    corr_reward_rows.append(float(reward))

                # 更新 RL 策略（仅在 warm-up 之后）
                if round_idx >= warmup_rounds:
                    client.update_rl_agent(reward, round_idx=round_idx)

                # 更新 EMA 基线
                client.expected_val_loss = (
                    (1 - client.loss_ema_eta) * client.expected_val_loss
                    + client.loss_ema_eta * loss_after_local
                )

                w_per_models[clnt_idx] = copy.deepcopy(trained_model)
                w_per_mods[clnt_idx] = copy.deepcopy(trained_mdl)

                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            # 3) 评估
            self._log_feature_reward_correlation(corr_feat_rows, corr_reward_rows, round_idx)
            self._test_on_all_clients(w_per_models, round_idx)

        # 训练结束后，再评估一次
        if getattr(self.args, "eval_benign_only", False):
            self._test_on_all_clients(w_per_models, self.args.comm_round - 1, client_indices=benign_ids)
        else:
            self._test_on_all_clients(w_per_models, self.args.comm_round - 1)

        # 收集最终 soft logits 并可视化
        self.final_round_slogits = []
        for client in self.client_list:
            _, client_slogits = client.extract_features_and_compute_soft_logits()
            self.final_round_slogits.append(copy.deepcopy(client_slogits))

        self.visualize_slogits()
        self._log_rl_statistics()
        self._log_final_personalized_acc_summary()


    def _log_rl_statistics(self):
        """记录 RL 训练统计（基于各客户端的 reward_history）"""
        total_rewards = 0.0
        total_clients = 0

        for client in self.client_list:
            if hasattr(client, "reward_history") and client.reward_history:
                total_rewards += float(sum(client.reward_history))
                total_clients += 1

        avg_reward = total_rewards / max(total_clients, 1)

        stats = {
            "total_training_rounds": self.args.comm_round,
            "average_reward_per_client": avg_reward,
            "final_performance": self.stat_info["global_generalized_acc"][-1]
            if self.stat_info["global_generalized_acc"]
            else 0.0,
        }

        self.logger.info("RL Training Statistics:")
        self.logger.info(stats)
