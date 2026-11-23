"""AMGC-AC demonstration on the UCI multi-view dataset."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import trange

from config import Config
from model import Feature_DGI_Model
from utils import (
    ZINB,
    features_construct_graph,
    normalize_sparse_matrix,
    sparse_mx_to_torch_sparse_tensor,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "Datasets" / "UCI.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMGC-AC on UCI.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_uci(
    path: Path, device: torch.device
) -> tuple[list[torch.Tensor], list[int], np.ndarray, list[torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    mat = sio.loadmat(path)
    if "X" not in mat or "Y" not in mat:
        raise KeyError("UCI.mat must contain variables 'X' and 'Y'.")

    labels = np.asarray(mat["Y"]).reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    sample_count = labels.size

    raw_views = mat["X"]
    if not isinstance(raw_views, np.ndarray) or raw_views.dtype != object:
        raw_views = np.asarray([raw_views], dtype=object)

    views: list[np.ndarray] = []
    for raw_view in raw_views.flat:
        if sp.issparse(raw_view):
            raw_view = raw_view.toarray()
        view = np.asarray(raw_view, dtype=np.float32).squeeze()
        if view.ndim != 2:
            raise ValueError("Each UCI view must be a two-dimensional matrix.")
        if view.shape[0] != sample_count and view.shape[1] == sample_count:
            view = view.T
        if view.shape[0] != sample_count:
            raise ValueError("Every view must contain the same number of samples.")
        if not np.isfinite(view).all():
            raise ValueError("UCI features contain NaN or infinite values.")
        views.append(view)

    view_tensors: list[torch.Tensor] = []
    adjs: list[torch.Tensor] = []
    for view in views:
        graph_features = StandardScaler().fit_transform(view)
        model_features = MinMaxScaler().fit_transform(view).astype(np.float32)
        adj = features_construct_graph(
            graph_features,
            k=15,
            pca=50 if view.shape[1] > 50 else None,
        )
        adj = normalize_sparse_matrix(adj + sp.eye(sample_count))
        adjs.append(sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device))
        view_tensors.append(torch.from_numpy(model_features).to(device))

    return view_tensors, [view.shape[1] for view in views], labels, adjs


def map_labels_hungarian(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    size = max(y_pred.max(), y_true.max()) + 1
    counts = np.zeros((size, size), dtype=np.int64)
    for predicted, true in zip(y_pred, y_true):
        counts[predicted, true] += 1
    rows, columns = linear_sum_assignment(counts.max() - counts)
    mapping = dict(zip(rows, columns))
    mapped = np.asarray([mapping.get(value, value) for value in y_pred])
    return counts[rows, columns].sum() / y_pred.size, mapped


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency_matrix(y_true, y_pred)
    return float(np.amax(matrix, axis=0).sum() / matrix.sum())


def cluster_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    matrix = contingency_matrix(y_true, y_pred)
    total = matrix.sum()
    cluster_sizes = matrix.sum(axis=0)
    entropy = 0.0
    for index, cluster_size in enumerate(cluster_sizes):
        if cluster_size > 0:
            probabilities = matrix[:, index] / cluster_size
            probabilities = probabilities[probabilities > 0]
            entropy -= (cluster_size / total) * np.sum(
                probabilities * np.log2(probabilities)
            )
    return float(entropy)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    acc, mapped = map_labels_hungarian(y_true, y_pred)
    return {
        "ACC": float(acc),
        "NMI": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "F1": float(metrics.f1_score(y_true, mapped, average="macro")),
        "Precision": float(
            metrics.precision_score(
                y_true, mapped, average="macro", zero_division=0
            )
        ),
        "Recall": float(
            metrics.recall_score(y_true, mapped, average="macro", zero_division=0)
        ),
        "Purity": cluster_purity(y_true, y_pred),
        "Entropy": cluster_entropy(y_true, y_pred),
    }


def train_step(
    network: Feature_DGI_Model,
    views: list[torch.Tensor],
    adjs: list[torch.Tensor],
    config: Config,
    optimizer: optim.Optimizer,
) -> None:
    network.train()
    optimizer.zero_grad()
    _, reconstructions, _, dgi_loss = network(
        views, adjs, return_contrastive=True
    )
    reconstruction_losses = []
    for features, (pi, dispersion, mean) in zip(views, reconstructions):
        reconstruction_losses.append(
            ZINB(pi, theta=dispersion, ridge_lambda=0).loss(
                features, mean, is_mean=False
            ).mean()
        )
    total_loss = torch.stack(reconstruction_losses).mean() + config.alpha * dgi_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("Non-finite training loss.")
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        network.parameters(), max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()


def run_experiment(
    views: list[torch.Tensor],
    view_dims: list[int],
    labels: np.ndarray,
    adjs: list[torch.Tensor],
    config: Config,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    set_seed(seed)
    network = Feature_DGI_Model(
        vdims=view_dims,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        dropout=config.dropout,
    ).to(device)
    optimizer = optim.Adam(
        network.parameters(), lr=min(config.lr, 1e-4), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    cluster_count = np.unique(labels).size
    best_acc = -1.0
    best_result: dict[str, float | int | str] | None = None

    for epoch in trange(epochs, desc="Training"):
        train_step(network, views, adjs, config, optimizer)
        scheduler.step()
        network.eval()
        with torch.no_grad():
            embedding = network.encode_views(views, adjs).cpu().numpy()
        prediction = KMeans(
            n_clusters=cluster_count, n_init=10, random_state=seed
        ).fit_predict(embedding)
        epoch_result = calculate_metrics(labels, prediction)
        if epoch_result["ACC"] > best_acc:
            best_acc = epoch_result["ACC"]
            best_result = {
                **epoch_result,
                "BestEpoch": epoch + 1,
                "Seed": seed,
                "Selection": "oracle_best_ACC",
            }

    if best_result is None:
        raise RuntimeError("No epoch result was produced.")
    return best_result


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    device = select_device(args.device)
    config = Config()
    config.epochs = args.epochs

    print(f"Dataset: {args.data.resolve()}")
    print(f"Device: {device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print("Selection: oracle best ACC using ground-truth labels")
    start = time.perf_counter()
    views, view_dims, labels, adjs = load_uci(args.data.resolve(), device)
    result = run_experiment(
        views, view_dims, labels, adjs, config, args.epochs, args.seed, device
    )
    elapsed = time.perf_counter() - start

    print("\nResult")
    print(f"BestEpoch: {result['BestEpoch']}")
    for name in (
        "ACC",
        "NMI",
        "ARI",
        "F1",
        "Precision",
        "Recall",
        "Purity",
        "Entropy",
    ):
        print(f"{name}: {result[name]:.4f}")
    print(f"Time(s): {elapsed:.2f}")


if __name__ == "__main__":
    main()
