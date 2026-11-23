import numpy as np
import torch


class Config(object):
    def __init__(self, config_file=None):
        """IMVE 模型的参数配置类

        去除了对外部 .ini 配置文件的依赖，直接在类内部完成参数初始化。
        """

        # ==========================================
        # 1. 基础数据参数 (Data Setting)
        # ==========================================
        # 模型的特征输入维度，在训练脚本中通常会根据数据集的特征数/基因数动态更新
        self.fdim = 3000

        # ==========================================
        # 2. 核心模型参数 (Model Setup)
        # ==========================================
        self.seed = 21  # 随机种子，确保实验可复现
        self.lr = 3e-4  # 学习率 (Learning Rate)
        self.weight_decay = 5e-4  # 权重衰减 (L2 正则化)
        self.nhid1 = 128  # 编码器第一隐藏层维度
        self.nhid2 = 64  # 编码器第二隐藏层/低维嵌入维度
        self.dropout = 0.1  # Dropout 概率
        self.epochs = 500  # 训练迭代轮数
        self.contrastive_dim = 128  # 对比学习特征投影维度

        # 设备与随机性控制
        self.no_cuda = False  # 是否禁用 GPU (False 表示优先使用 CUDA)
        self.no_seed = False  # 是否禁用固定随机种子

        # ==========================================
        # 3. 损失函数权重控制 (Loss Weight)
        # ==========================================
        # L = L_ZINB + alpha * L_DGI; alpha weights the DGI term.

        self.alpha = 0.01


if __name__ == "__main__":
    # 测试初始化
    cfg = Config()
    print("Config 配置加载成功：")
    print(
        f"Input Dim: {cfg.fdim}, LR: {cfg.lr}, Epochs: {cfg.epochs}, Seed: {cfg.seed}"
    )
