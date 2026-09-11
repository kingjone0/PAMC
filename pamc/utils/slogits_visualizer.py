import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional
import torch


class SimpleSoftLogitsVisualizer:
    """
    简化的软logits可视化工具，只为每个客户端生成柱状图
    """

    def __init__(self, output_dir: str = "./slogits_plots"):
        """
        初始化可视化器

        Args:
            output_dir: 图片保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_client_slogits(self,
                            slogits: torch.Tensor,
                            client_idx: int,
                            round_idx: int,
                            num_classes: Optional[int] = None):
        """
        为单个客户端绘制软logits柱状图

        Args:
            slogits: 软logits张量 [num_classes]
            client_idx: 客户端索引
            round_idx: 轮次索引
            num_classes: 类别数量，如果为None则自动推断
        """
        # 转换为numpy数组
        if torch.is_tensor(slogits):
            slogits_np = slogits.detach().cpu().numpy()
        else:
            slogits_np = np.array(slogits)

        # 如果是一维数组，直接使用
        if slogits_np.ndim == 1:
            values = slogits_np
            classes = range(len(values))
        else:
            # 如果是二维或多维，展平
            values = slogits_np.flatten()
            classes = range(len(values))

        # 创建图形
        plt.figure(figsize=(10, 6))

        # 绘制柱状图
        bars = plt.bar(classes, values, alpha=0.7, color='skyblue', edgecolor='black')

        # 添加数值标签（如果类别数不多）
        if len(values) <= 20:
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 设置图形属性
        plt.xlabel('Class Index')
        plt.ylabel('Soft Logits Value')
        plt.title(f'Client {client_idx} - Soft Logits Distribution (Round {round_idx})')
        plt.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        filename = os.path.join(self.output_dir, f"client_{client_idx}_round_{round_idx}_slogits.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved soft logits plot for client {client_idx}: {filename}")

        return filename

    def plot_all_clients(self,
                         w_per_slogits: List[torch.Tensor],
                         round_idx: int):
        """
        为所有客户端绘制软logits柱状图

        Args:
            w_per_slogits: 各客户端的软logits列表
            round_idx: 轮次索引
        """
        saved_files = []

        for client_idx, slogits in enumerate(w_per_slogits):
            filename = self.plot_client_slogits(slogits, client_idx, round_idx)
            saved_files.append(filename)

        # 生成汇总信息
        summary_file = os.path.join(self.output_dir, f"round_{round_idx}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Soft Logits Visualization Summary - Round {round_idx}\n")
            f.write("=" * 50 + "\n")
            for i, filepath in enumerate(saved_files):
                f.write(f"Client {i}: {filepath}\n")

        print(f"Summary file saved: {summary_file}")

        return saved_files


def visualize_slogits_simple(w_per_slogits: List[torch.Tensor],
                             round_idx: int,
                             output_dir: str = "./slogits_plots"):
    """
    便捷函数：简单可视化所有客户端的软logits

    Args:
        w_per_slogits: 各客户端的软logits列表
        round_idx: 轮次索引
        output_dir: 输出目录

    Returns:
        list: 保存的图片文件路径列表
    """
    visualizer = SimpleSoftLogitsVisualizer(output_dir)
    return visualizer.plot_all_clients(w_per_slogits, round_idx)


# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_slogits = [
        torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3]),  # 客户端0，5个类别
        torch.tensor([0.05, 0.35, 0.1, 0.25, 0.25]),  # 客户端1
        torch.tensor([0.2, 0.1, 0.3, 0.15, 0.25]),  # 客户端2
    ]

    # 可视化
    saved_files = visualize_slogits_simple(test_slogits, round_idx=99)
    print(f"Generated {len(saved_files)} plots")