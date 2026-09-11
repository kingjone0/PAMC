# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/24 9:43
@Auth ： 康锦程
"""
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pamc.models.vgg import vgg11
from pamc.models.lenet5 import LeNet5
from pamc.models.cnn import cnn_cifar10, cnn_cifar100, simplecnn
from pamc.data.dataprocess.cifar10.cifar10_trans import cifar10_dataset_read
from pamc.data.dataprocess.cifar100.cifar100_trans import cifar100_dataset_read
from pamc.data.dataprocess.mnist.mnist_trans import mnist_dataset_read
from pamc.data.dataprocess.emnist.emnist_trans import emnist_dataset_read
from pamc.algorithm.pamc import BACAPI, RLEnhancedBACAPI
from pamc.algorithm.dfl_trainer import MyModelTrainer


def logger_config(log_path, logging_name):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='cnn_cifar10', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0, metavar='N',
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='noniid', metavar='N',
                        help="current supporting two types of data partition, one called 'noniid' short for Dirichlet"
                             "one called 'pat' short for how many classes allocated for each client")

    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--partition_save_path', type=str, default=None, metavar='N')

    parser.add_argument('--partition_load_path', type=str, default='data/cifar10/cn10_alpha_0.1.json', metavar='N')

    # RL相关参数
    parser.add_argument('--use_rl', type=str2bool, nargs='?', const=True, default=True,
                        help='是否使用RL增强的客户端选择')

    parser.add_argument('--rl_algorithm', type=str, default='ppo', choices=['ppo', 'reinforce'],
                        help='RL算法选择（目前实现为连续动作PPO）')

    parser.add_argument('--policy_lr', type=float, default=3e-4,
                        help='策略网络学习率（会映射到 rl_lr）')
    # 3e-4
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='熵正则化系数（当前实现中固定在RL内部使用，可作为记录用）')

    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO裁剪参数（会映射到 rl_clip_epsilon）')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子（会映射到 rl_gamma）')

    parser.add_argument('--temperature', type=float, default=2.0, metavar='TP',
                        help='temperature parameters when extracting soft logits')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--frac', type=float, default=0.2, metavar='NN',
                        help='selection fraction each round')

    parser.add_argument('--trust_region_tau', type=float, default=0.8, metavar='Tau')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--num_components", type=int, default=10,
                        help='number of PCA components')

    parser.add_argument("--expected_acc", type=float, default=0.9,
                        help='expected accuracy of the model')

    parser.add_argument("--bandit_alpha", type=float, default=2.0,
                        help='bandit_alpha of rlAgent（非RL基线用）')

    parser.add_argument("--warmup_rounds", type=int, default=10,
                        help="number of initial rounds using random client selection before RL agent takes over")

    # 新增/预留：RL隐藏层维度、内部名称对应
    parser.add_argument("--rl_hidden_dim", type=int, default=256,
                        help="hidden dimension of RL policy network")

    parser.add_argument("--feature_mode", type=str, default="all",
                        choices=["all", "sim", "feat", "logits"],
                        help="neighbor feature mode for RL state")

    parser.add_argument("--byzantine_ratio", type=float, default=0.0,
                        help="fraction of malicious clients")
    parser.add_argument("--byzantine_clients", type=str, default="",
                        help="comma-separated malicious client ids, e.g., 0,3,5")
    parser.add_argument("--attack_type", type=str, default="sign_flip",
                        choices=["inv_grad", "shuffle", "same_value", "sign_flip", "gauss"],
                        help="byzantine attack type")
    parser.add_argument("--eval_benign_only", action="store_true", default=True,
                        help="evaluate only benign clients")

    parser.add_argument("--rl_rollout_len", type=int, default=5,
                        help="number of steps to collect before PPO update")

    parser.add_argument("--heuristic_sim_weight", type=float, default=0.2,
                        help="heuristic aggregation weight for model similarity")
    parser.add_argument("--heuristic_feat_weight", type=float, default=0.4,
                        help="heuristic aggregation weight for feature complementarity")
    parser.add_argument("--heuristic_logit_weight", type=float, default=0.4,
                        help="heuristic aggregation weight for logits complementarity")

    parser.add_argument("--reward_agg_weight", type=float, default=1.0,
                        help="reward weight for post-aggregation validation gain")
    parser.add_argument("--reward_local_weight", type=float, default=0.0,
                        help="reward weight for post-local-training validation gain")
    parser.add_argument("--reward_acc_weight", type=float, default=0.0,
                        help="reward bonus weight for local accuracy improvement")
    parser.add_argument("--reward_metric_weight", type=float, default=0.1,
                        help="reward bonus weight for selected-neighbor metric quality")
    parser.add_argument("--reward_clip", type=float, default=2.0,
                        help="clip range for normalized PPO rewards")
    parser.add_argument("--reward_norm_beta", type=float, default=0.05,
                        help="EMA factor for reward normalization")

    parser.add_argument("--ppo_update_epochs", type=int, default=4,
                        help="number of PPO epochs per rollout update")
    parser.add_argument("--ppo_value_coef", type=float, default=0.5,
                        help="value loss coefficient in PPO")
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01,
                        help="entropy regularization coefficient in PPO")
    parser.add_argument("--ppo_weight_prior_blend", type=float, default=0.0,
                        help="blend ratio between heuristic metric prior and PPO metric weights")
    parser.add_argument("--ppo_dirichlet_scale", type=float, default=10.0,
                        help="concentration scaling for Dirichlet metric-weight policy")
    parser.add_argument("--staleness_decay", type=float, default=0.05,
                        help="decay factor applied to cached neighbor metric scores")
    parser.add_argument("--refresh_bonus", type=float, default=0.005,
                        help="small exploration bonus for stale neighbor cache entries")
    parser.add_argument("--staleness_cap", type=float, default=20.0,
                        help="maximum staleness used in state normalization and scoring")
    parser.add_argument("--neighbor_sample_mode", type=str, default="prob",
                        choices=["prob", "topk"],
                        help="communication-neighbor selection mode after scoring")
    parser.add_argument("--neighbor_sample_tau", type=float, default=0.3,
                        help="softmax temperature for probabilistic communication-neighbor sampling")
    parser.add_argument("--neighbor_sample_tau_min", type=float, default=0.03,
                        help="minimum softmax temperature after communication-neighbor sampling annealing")

    return parser


def load_data(args, dataset_name, logger):
    if dataset_name == "cifar10":
        args.num_classes = 10
        args.data_dir += "/cifar10"
        train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions = cifar10_dataset_read(
            args.data_dir, args.batch_size, args.client_num_in_total, args.partition_method,
            args.partition_alpha, logger, args.partition_save_path, args.partition_load_path)
    elif dataset_name == "cifar100":
        args.num_classes = 100
        args.data_dir += "/cifar100"
        train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions = cifar100_dataset_read(
            args.data_dir, args.batch_size, args.client_num_in_total, args.partition_method,
            args.partition_alpha, logger, args.partition_save_path, args.partition_load_path)
    elif dataset_name == "mnist":
        args.num_classes = 10
        args.data_dir += "/mnist"
        train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions = mnist_dataset_read(
            args.data_dir, args.batch_size, args.client_num_in_total, args.partition_method,
            args.partition_alpha, logger, args.partition_save_path, args.partition_load_path)
    elif dataset_name == "emnist":
        args.num_classes = 47
        args.data_dir += "/emnist"
        train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions = emnist_dataset_read(
            args.data_dir, args.batch_size, args.client_num_in_total, args.partition_method,
            args.partition_alpha, logger, args.partition_save_path, args.partition_load_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = [train_dataloaders, val_dataloaders, test_loader, data_local_num_dict, data_distributions]
    return dataset


def create_model(args, model_name, class_num):
    model = None
    if model_name == "lenet5":
        model = LeNet5(class_num)
    elif model_name == "cnn_cifar10":
        model = simplecnn(84, 10)
    elif model_name == "cnn_cifar100":
        model = cnn_cifar100()
    elif model_name == "vgg11":
        model = vgg11(class_num)
    return model


def custom_model_trainer(args, model):
    return MyModelTrainer(model, args)


def setup_rl_training(args, dataset, device, model_trainer, logger):
    """设置RL训练环境：根据 use_rl 选择基线或RL增强版本"""
    if args.use_rl:
        logger.info("使用RL增强的联邦学习训练（RLEnhancedBACAPI）")
        trainer = RLEnhancedBACAPI(dataset, device, args, model_trainer, logger)
    else:
        logger.info("使用基础联邦学习训练（BACAPI）")
        trainer = BACAPI(dataset, device, args, model_trainer, logger)

    return trainer


if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description="PAMC decentralized federated learning experiment"))
    args = parser.parse_args()
    print("torch version {}".format(torch.__version__))

    device = torch.device("cuda:" + str(args.gpu))
    print(device)

    args.data_dir = os.path.abspath(args.data_dir)

    # -----------------------------
    # 映射/补全 RL 内部使用的超参数名称
    # -----------------------------
    # client.py / rlAgent.py 中使用的是：
    #   rl_lr, rl_gamma, rl_clip_epsilon, rl_hidden_dim
    # 这里用 main 脚本中的 policy_lr / gamma / clip_epsilon 映射过去
    args.rl_lr = getattr(args, "rl_lr", args.policy_lr)
    args.rl_gamma = getattr(args, "rl_gamma", args.gamma)
    args.rl_clip_epsilon = getattr(args, "rl_clip_epsilon", args.clip_epsilon)
    # rl_hidden_dim 已在 add_args 中定义

    data_partition = args.partition_method
    data_partition += str(args.partition_alpha)

    # 更新身份标识以包含RL信息
    args.identity = "PAMC" + "-" + args.dataset + "-" + data_partition
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    args.identity += "-mdl" + args.model
    args.identity += "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-bc" + str(args.byzantine_clients)

    if args.use_rl:
        args.identity += "-RL_" + args.rl_algorithm
        args.identity += "-lr" + str(args.policy_lr)
        args.identity += "-ent" + str(args.entropy_coef)
    else:
        args.identity += "-bandit_alpha" + str(args.bandit_alpha)

    args.identity += '-fm' + str(args.feature_mode)
    args.identity += "-sd" + str(args.staleness_decay)
    args.identity += "-rb" + str(args.refresh_bonus)
    args.identity += "-sc" + str(args.staleness_cap)
    args.identity += "-nsm" + str(args.neighbor_sample_mode)
    args.identity += "-nst" + str(args.neighbor_sample_tau)
    args.identity += "-nstmin" + str(args.neighbor_sample_tau_min)
    args.identity += "-ea" + str(args.expected_acc)
    args.identity += "-bycl" + str(args.byzantine_clients)
    args.identity += '-seed' + str(args.seed)
    if getattr(args, "tag", ""):
        args.identity += "-tag" + str(args.tag)

    cur_dir = os.path.abspath(__file__).rsplit(os.sep, 1)[0]
    log_dir = os.path.join(cur_dir, 'LOG', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, args.identity + '.log')
    logger = logger_config(log_path=log_path, logging_name=args.identity)

    logger.info(args)
    logger.info("running at device {}".format(device))

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 加载数据
    logger.info("加载数据集...")
    dataset = load_data(args, args.dataset, logger)

    # 创建模型
    logger.info("创建模型...")
    # dataset[-1] 是 data_distributions，dataset[-1][0] 是某个客户端的分布向量
    num_classes = len(dataset[-1][0])
    model = create_model(args, model_name=args.model, class_num=num_classes)
    model_trainer = custom_model_trainer(args, model)
    logger.info(model)

    # 设置训练器
    logger.info("初始化训练器...")
    trainer = setup_rl_training(args, dataset, device, model_trainer, logger)

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    logger.info("训练完成!")
