import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class FedResNet_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 10

        # 特征提取器 (清晰划分卷积和残差块)
        self.feature_extractor = nn.Sequential(
            # 输入适配层
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 残差块组
            self._make_layer(64, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),

            # 输出处理
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 分类器 (保持与原设计相同的层次结构)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_classes)
        )

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """优化的残差层构建方法"""
        downsample = None
        expansion = BasicBlock.expansion  # expansion=1

        # 维度修正条件判断
        if stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class FedResNet_CIFAR100(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 100

        # 特征提取器 (包含所有卷积和残差块)
        self.feature_extractor = nn.Sequential(
            # 输入层
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 残差块组
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),

            # 输出处理
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 分类器 (独立模块)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.n_classes)
        )

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """残差块构建方法（带维度修正）"""
        downsample = None
        expansion = BasicBlock.expansion  # 值为1

        # 维度修正条件判断
        if stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def test_models():
    # 定义模型及其配置
    model_configs = [
        {  # CIFAR10配置
            'model_class': FedResNet_CIFAR10,
            'n_classes': 10,
            'feature_dim': 256,
            'res_blocks': [
                (3, 64, 64, 32, 32),  # (layer_idx, in_ch, out_ch, h, w)
                (4, 64, 128, 32, 32),
                (5, 128, 256, 16, 16)
            ]
        },
        {  # CIFAR100配置
            'model_class': FedResNet_CIFAR100,
            'n_classes': 100,
            'feature_dim': 512,
            'res_blocks': [
                (3, 64, 64, 32, 32),
                (4, 64, 128, 32, 32),
                (5, 128, 256, 16, 16),
                (6, 256, 512, 8, 8)
            ]
        }
    ]

    for config in model_configs:
        model_class = config['model_class']
        n_classes = config['n_classes']
        feature_dim = config['feature_dim']
        res_blocks = config['res_blocks']

        print(f"\n=== 测试模型: {model_class.__name__} ===")
        model = model_class()

        # 测试数据
        input_tensor = torch.randn(2, 3, 32, 32)

        # 完整模型前向测试
        output = model(input_tensor)
        assert output.shape == (2, n_classes), (
            f"[{model_class.__name__}] 输出维度错误: {output.shape} vs (2, {n_classes})"
        )

        # 特征提取器测试
        features = model.feature_extractor(input_tensor)
        assert features.shape == (2, feature_dim), (
            f"[{model_class.__name__}] 特征提取器错误: {features.shape} vs (2, {feature_dim})"
        )

        # 分类器测试
        classifier_input = torch.randn(2, feature_dim)
        logits = model.classifier(classifier_input)
        assert logits.shape == (2, n_classes), (
            f"[{model_class.__name__}] 分类器错误: {logits.shape} vs (2, {n_classes})"
        )

        # 残差块维度测试
        print(f"残差块测试:")
        for layer_idx, in_ch, out_ch, h, w in res_blocks:
            block = model.feature_extractor[layer_idx][0]
            test_input = torch.randn(2, in_ch, h, w)

            # 计算预期输出尺寸
            stride = block.conv1.stride
            expected_h = h // stride[0]
            expected_w = w // stride[1]
            expected_shape = (2, out_ch, expected_h, expected_w)

            # 执行前向传播
            output = block(test_input)

            # 断言维度
            assert output.shape == expected_shape, (
                f"[{model_class.__name__}] 残差块(layer{layer_idx})维度错误\n"
                f"期望: {expected_shape}\n"
                f"实际: {output.shape}"
            )

        print(f"{model_class.__name__} 所有测试通过！\n")


if __name__ == "__main__":
    test_models()