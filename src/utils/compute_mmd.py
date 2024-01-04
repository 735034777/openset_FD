import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def compute_mmd(Xs, Xt, gamma=None):
    """
    计算最大均值差异（MMD）。

    参数:
    Xs -- 源域样本的特征矩阵。
    Xt -- 目标域样本的特征矩阵。
    gamma -- RBF核的参数。如果为None，则自动计算。

    返回:
    MMD值。
    """

    # 计算 RBF 核的参数 gamma，如果没有指定
    if gamma is None:
        gamma = 1.0 / (Xs.shape[1] * np.var(np.concatenate((Xs, Xt))))

    # 计算核矩阵
    Kss = rbf_kernel(Xs, Xs, gamma)
    Ktt = rbf_kernel(Xt, Xt, gamma)
    Kst = rbf_kernel(Xs, Xt, gamma)

    # 计算 MMD
    mmd = np.mean(Kss) + np.mean(Ktt) - 2 * np.mean(Kst)
    return mmd


# 示例数据（随机生成）
np.random.seed(0)
Xs = np.random.randn(100, 10)  # 源域样本
Xt = np.random.randn(80, 10)  # 目标域样本

# 计算 MMD
mmd_value = compute_mmd(Xs, Xt)
mmd_value
