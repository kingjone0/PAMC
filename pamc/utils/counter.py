# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/5 15:43
@Auth ： 康锦程
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


def print_model_param_nums(model=None):
    if model is None:
        model = torchvision.models.alexnet()
    total = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('  + Number of params: %.2f' % total)


def count_training_flops(model, dataset, full=False):
    return 3 * count_model_param_flops(model, dataset, full=full)


def count_inference_flops(model, dataset):
    return count_model_param_flops(model, dataset)


def register_hooks(model, hooks):
    handles = []

    def apply_hooks(net):
        if isinstance(net, nn.Conv2d):
            handles.append(net.register_forward_hook(hooks['conv']))
        elif isinstance(net, nn.Linear):
            handles.append(net.register_forward_hook(hooks['linear']))
        elif isinstance(net, nn.BatchNorm2d):
            handles.append(net.register_forward_hook(hooks['bn']))
        elif isinstance(net, nn.ReLU):
            handles.append(net.register_forward_hook(hooks['relu']))
        elif isinstance(net, (nn.MaxPool2d, nn.AvgPool2d)):
            handles.append(net.register_forward_hook(hooks['pooling']))
        elif isinstance(net, nn.Upsample):
            handles.append(net.register_forward_hook(hooks['upsample']))
        for child in net.children():
            apply_hooks(child)

    apply_hooks(model)
    return handles


def count_model_param_flops(model=None, dataset=None, multiply_adds=True, full=False):
    if model is None:
        model = torchvision.models.alexnet()

    flops_dict = {
        'conv': [],
        'linear': [],
        'bn': [],
        'relu': [],
        'pooling': [],
        'upsample': []
    }

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        num_weight_params = torch.numel(self.weight.data) if full else (self.weight.data != 0).float().sum()
        flops = (num_weight_params * (
            2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
        flops_dict['conv'].append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = torch.numel(self.weight.data) if full else (self.weight.data != 0).float().sum()
        bias_ops = torch.numel(self.bias.data) if self.bias is not None else 0
        flops = batch_size * (weight_ops * (2 if multiply_adds else 1) + bias_ops)
        flops_dict['linear'].append(flops)

    def bn_hook(self, input, output):
        flops_dict['bn'].append(input[0].nelement() * 2)

    def relu_hook(self, input, output):
        flops_dict['relu'].append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size * self.kernel_size
        flops = kernel_ops * output_height * output_width * batch_size
        flops_dict['pooling'].append(flops)

    def upsample_hook(self, input, output):
        batch_size, input_channels, output_height, output_width = output[0].size()
        flops = output_height * output_width * batch_size * 12
        flops_dict['upsample'].append(flops)

    hooks = {
        'conv': conv_hook,
        'linear': linear_hook,
        'bn': bn_hook,
        'relu': relu_hook,
        'pooling': pooling_hook,
        'upsample': upsample_hook
    }

    handles = register_hooks(model, hooks)

    if dataset == "emnist":
        input_channel, input_res = 1, 28
    elif dataset in ["cifar10", "cifar100"]:
        input_channel, input_res = 3, 32
    elif dataset == "tiny":
        input_channel, input_res = 3, 64
    else:
        raise ValueError("Unsupported dataset")

    device = next(model.parameters()).device
    input = Variable(torch.rand(1, input_channel, input_res, input_res), requires_grad=True).to(device)
    model(input)

    total_flops = sum(sum(flops) for flops in flops_dict.values())
    for handle in handles:
        handle.remove()

    return total_flops
