import os
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

# -----------------------------
# 基础张量工具
# -----------------------------
def _nan2zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

# -----------------------------
# 特征图构建 (保留核心特征多指标融合方法)
# -----------------------------
def _row_wise_min_max(M: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
    M_norm = M.copy()
    n = M.shape[0]
    for i in range(n):
        idx = np.where(neighbor_mask[i])[0]
        if idx.size == 0:
            continue
        vals = M[i, idx]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            M_norm[i, idx] = (vals - vmin) / (vmax - vmin)
        else:
            M_norm[i, idx] = 0.0
    return M_norm


def _spearman_similarity_np(features: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(features, axis=0), axis=0) + 1
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    mat = (centered @ centered.T) / (norms @ norms.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def features_construct_graph(X, k=15, pca=50):
    """
    超大样本单细胞多组学专用：分批近邻稀疏建图（彻底杜绝 3.5 GiB 内存溢出）
    支持 N > 30000 的高维单细胞流形构建。
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    N = X.shape[0]
    print(f" -> 开始构建流形图，样本数 N = {N}, 特征维 = {X.shape[1]}")

    # 1. 自动高维过滤：如果特征特别多且开启了 PCA，先做基础降维平滑噪点
    if pca is not None and X.shape[1] > pca:
        pca_model = PCA(n_components=pca, random_state=42)
        X_encoded = pca_model.fit_transform(X)
    else:
        X_encoded = X

    # 2. 【核心优化】使用 sklearn 的 NearestNeighbors 寻找每个细胞最临近的 k 个邻居
    # 这样只会生成包含 N * k 个元素的稀疏结构，不再实例化 N * N 的 3.5GB 恐怖稠密矩阵
    # 针对超大样本，内部使用高效的 'ball_tree' 或 'kd_tree' 算法实现秒级检索
    print(f" -> 正在通过近邻搜索算法，计算每个细胞的 Top-{k} 临近流形...")

    # 因为需要扣除自身，所以找 k+1 个邻居
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nn.fit(X_encoded)

    # distances: 对应的余弦距离，indices: 邻居的索引索引
    distances, indices = nn.kneighbors(X_encoded)

    # 3. 将距离转化为相似度（余弦相似度 = 1 - 余弦距离）
    # 只针对这选出的 k 个近邻赋值，其余位置保持为 0，完美契合稀疏图
    weights = 1.0 - distances

    # 4. 构造高效的 CSR 格式稀疏邻接矩阵
    row_indices = np.repeat(np.arange(N), k + 1)
    col_indices = indices.flatten()
    data_values = weights.flatten()

    # 组装初始稀疏矩阵
    adj_sparse = sp.csr_matrix((data_values, (row_indices, col_indices)), shape=(N, N))

    # 5. 保证图的对称性：A = (A + A.T) / 2 （稀疏矩阵运算，极度节省内存）
    print(" -> 正在执行稀疏流形对称化校准...")
    adj_sparse = (adj_sparse + adj_sparse.T) / 2.0

    # 6. 抹除自环：保证细胞自己到自己的连接在这一阶段为 0
    adj_sparse = sp.lil_matrix(adj_sparse)
    adj_sparse.setdiag(0)

    print(f" -> 稀疏邻接矩阵建图成功！有效非零边数量: {adj_sparse.nnz}")
    return adj_sparse.tocsr()

# -----------------------------
# 稀疏工具
# -----------------------------
def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_sparse_matrix(mx: sp.spmatrix) -> sp.spmatrix:
    """Apply the row normalization D^{-1} A used by the previous model."""
    rowsum = np.asarray(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    r_inv[nonzero] = np.power(rowsum[nonzero], -1.0)
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def dopca(data: np.ndarray, dim: int = 50) -> np.ndarray:
    return PCA(n_components=dim).fit_transform(data)

# -----------------------------
# ZINB 损失计算
# -----------------------------
class NB(object):
    def __init__(self, theta=None, scale_factor: float = 1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, mean: bool = True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6, device=y_true.device, dtype=y_true.dtype))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + self.eps)) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        )
        final = t1 + t2
        final = _nan2inf(final)
        return torch.mean(final) if mean else final


class ZINB(object):
    def __init__(self, pi, theta, ridge_lambda=0.0, eps=1e-8):
        self.pi = pi
        self.theta = theta
        self.ridge_lambda = ridge_lambda
        self.eps = eps

    def loss(self, x, mean, is_mean=False):
        eps = self.eps
        # 1. 强力约束 mean 和 theta 的边界，防止 log(0) 或数值爆炸
        mean = torch.clamp(mean, min=1e-5, max=1e6)
        theta = torch.clamp(self.theta, min=1e-4, max=1e6)
        pi = torch.clamp(self.pi, min=1e-6, max=1.0 - 1e-6)

        # 2. NB 部分计算
        t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
        t2 = (theta + x) * torch.log(1.0 + mean / (theta + eps)) + x * (
            torch.log(theta + eps) - torch.log(mean + eps)
        )
        nb_final = t1 + t2
        del t1, t2

        # 3. 避免 nb_final 过大导致 exp(-nb_final) 溢出，对 nb_final 进行安全裁剪
        nb_final_clamped = torch.clamp(nb_final, min=-80.0, max=80.0)

        # 4. 零膨胀概率计算
        zero_nb = torch.log(pi + (1.0 - pi) * torch.exp(-nb_final_clamped) + eps)
        nonzero_nb = torch.log(1.0 - pi + eps) - nb_final_clamped

        # 5. 根据真实输入是否为 0 选择 loss
        is_zero = (x < 1e-5)
        zinb_loss_matrix = torch.where(is_zero, -zero_nb, -nonzero_nb)

        # 🔴【关键修复】对最终 Loss 矩阵中的 -inf/inf/nan 强制清零或截断
        zinb_loss_matrix = torch.nan_to_num(zinb_loss_matrix, nan=0.0, posinf=1e4, neginf=0.0)

        if self.ridge_lambda > 0:
            ridge = self.ridge_lambda * torch.sum(torch.square(pi))
            zinb_loss_matrix = zinb_loss_matrix + ridge

        if is_mean:
            return torch.mean(zinb_loss_matrix)

        return zinb_loss_matrix
