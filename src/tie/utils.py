import math

import numpy as np
import pandas as pd
import torch
from mitreattack.stix20 import MitreAttackData

from .constants import PredictionMethod


def get_mitre_technique_ids_to_names(stix_filepath: str) -> dict[str, str]:
    """Gets all MITRE technique ids mapped to their description."""
    mitre_attack_data = MitreAttackData(stix_filepath)
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)

    all_technique_ids = {}

    for technique in techniques:
        external_references = technique.get("external_references")
        mitre_references = tuple(
            filter(
                lambda external_reference: external_reference.get("source_name")
                == "mitre-attack",
                external_references,
            )
        )
        assert len(mitre_references) == 1
        mitre_technique_id = mitre_references[0]["external_id"]
        all_technique_ids[mitre_technique_id] = technique.get("name")

    return all_technique_ids


def _ensure_tensor(matrix) -> torch.Tensor:
    """Converts pandas/numpy inputs to a float tensor on the CPU."""

    if torch.is_tensor(matrix):
        tensor = matrix.to(dtype=torch.float32)
    elif isinstance(matrix, pd.DataFrame):
        tensor = torch.as_tensor(matrix.to_numpy(copy=False), dtype=torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(matrix), dtype=torch.float32)
    return tensor


def precision_at_k(predictions: pd.DataFrame, test_data: pd.DataFrame, k: int) -> float:
    """Calculates precision via the tensor variant (keeps original signature)."""

    predictions_tensor = _ensure_tensor(predictions)
    test_tensor = _ensure_tensor(test_data)
    return precision_at_k_tensor(predictions_tensor, test_tensor, k)


def recall_at_k(predictions: pd.DataFrame, test_data: pd.DataFrame, k: int) -> float:
    """Calculates recall via the tensor variant (keeps original signature)."""

    predictions_tensor = _ensure_tensor(predictions)
    test_tensor = _ensure_tensor(test_data)
    return recall_at_k_tensor(predictions_tensor, test_tensor, k)


def normalized_discounted_cumulative_gain(
    predictions: pd.DataFrame, test_data: pd.DataFrame, k: int = 10
) -> float:
    """Calculates NDCG via the tensor variant (keeps original signature)."""

    predictions_tensor = _ensure_tensor(predictions)
    test_tensor = _ensure_tensor(test_data)
    return normalized_discounted_cumulative_gain_tensor(predictions_tensor, test_tensor, k)


def precision_at_k_tensor(predictions: torch.Tensor, test_data: torch.Tensor, k: int) -> float:
    """Calculates precision directly on tensors."""

    if predictions.ndim != 2 or test_data.ndim != 2:
        raise ValueError("Predictions and test data must be 2D tensors.")
    if predictions.shape != test_data.shape:
        raise ValueError("Predictions and test data must have the same shape.")
    m, n = predictions.shape
    if not (0 < k <= n):
        raise ValueError("k must be between 1 and the number of items.")

    topk_indices = torch.topk(predictions, k, dim=1).indices
    mask = torch.zeros_like(predictions, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    hits = ((test_data > 0) & mask).sum(dim=1, dtype=torch.float32)
    return (hits.mean().item() / k)


def recall_at_k_tensor(predictions: torch.Tensor, test_data: torch.Tensor, k: int) -> float:
    """Calculates recall directly on tensors."""

    if predictions.ndim != 2 or test_data.ndim != 2:
        raise ValueError("Predictions and test data must be 2D tensors.")
    if predictions.shape != test_data.shape:
        raise ValueError("Predictions and test data must have the same shape.")
    m, n = predictions.shape
    if not (0 < k <= n):
        raise ValueError("k must be between 1 and the number of items.")

    topk_values = torch.topk(predictions, k, dim=1).values
    kth_values = topk_values[:, -1].unsqueeze(1)
    greater_mask = predictions > kth_values
    mask = greater_mask.clone()
    tie_mask = predictions == kth_values
    num_greater = greater_mask.sum(dim=1)
    tie_slots = (k - num_greater).clamp(min=0)
    tie_size = tie_mask.sum(dim=1)
    tie_fits = tie_size <= tie_slots
    tie_mask = tie_mask & tie_fits.unsqueeze(1)
    mask |= tie_mask
    hits = ((test_data > 0) & mask).sum(dim=1, dtype=torch.float32)
    num_test_items = (test_data > 0).sum(dim=1, dtype=torch.float32)
    valid = num_test_items > 0
    if not valid.any():
        return float("nan")
    recall_per_user = torch.zeros_like(num_test_items)
    recall_per_user[valid] = hits[valid] / num_test_items[valid]
    return recall_per_user[valid].mean().item()


def normalized_discounted_cumulative_gain_tensor(
    predictions: torch.Tensor, test_data: torch.Tensor, k: int = 10
) -> float:
    """Calculates NDCG directly on tensors."""

    if predictions.ndim != 2 or test_data.ndim != 2:
        raise ValueError("Predictions and test data must be 2D tensors.")
    if predictions.shape != test_data.shape:
        raise ValueError("Predictions and test data must have the same shape.")
    m, n = predictions.shape
    if not (0 < k <= n):
        raise ValueError("k must be between 1 and the number of items.")

    device = predictions.device
    predictions = predictions.to(dtype=torch.float64)
    test_data = test_data.to(dtype=torch.float64)
    _, sorted_indices = torch.sort(predictions, dim=1, descending=True)
    ranks = torch.arange(1, n + 1, device=device, dtype=torch.float64).unsqueeze(0).expand(m, -1)
    rank_positions = torch.zeros_like(predictions, dtype=torch.float64)
    rank_positions.scatter_(1, sorted_indices, ranks)
    numerator_mask = (rank_positions <= k) & (test_data > 0)
    denominator = torch.log2(rank_positions + 1)
    dcg_values = torch.zeros_like(predictions, dtype=torch.float64)
    dcg_values[numerator_mask] = 1.0 / denominator[numerator_mask]
    entity_dcg = dcg_values.sum(dim=1)

    test_set_size = (test_data > 0).sum(dim=1, dtype=torch.int32)
    user_idcg = torch.tensor(
        [_max_idcg(int(size.item()), k) for size in test_set_size],
        dtype=torch.float64,
        device=device,
    )
    valid_idcg = user_idcg > 0
    if not valid_idcg.any():
        return float("nan")
    valid_dcg = entity_dcg > 0
    if not valid_dcg.any():
        return float("nan")
    dcg_mean = entity_dcg[valid_dcg].mean()
    idcg_mean = user_idcg[valid_idcg].mean()
    if idcg_mean == 0:
        return float("nan")
    return (dcg_mean / idcg_mean).item()


def _max_idcg(test_set_size: int, k: int) -> float:
    """Computes the maximum DCG for the given test set size."""

    limit = max(0, min(test_set_size, k))
    return sum(1.0 / math.log2(i + 1) for i in range(1, limit + 1))


def _normalize_tensor(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    norm = torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True)
    norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
    return tensor / norm


def calculate_predicted_matrix(
    U, V, method: PredictionMethod = PredictionMethod.DOT
) -> torch.Tensor | np.ndarray:
    """Calculates the prediction matrix UV^T according to the dot or cosine product."""
    if torch.is_tensor(U) or torch.is_tensor(V):
        U_tensor = U if torch.is_tensor(U) else torch.tensor(U, dtype=torch.float32)
        V_tensor = V if torch.is_tensor(V) else torch.tensor(V, dtype=torch.float32)
        if U_tensor.dtype != torch.float32:
            U_tensor = U_tensor.float()
        if V_tensor.dtype != torch.float32:
            V_tensor = V_tensor.float()

        if method == PredictionMethod.DOT:
            result = torch.matmul(U_tensor, V_tensor.t())
        elif method == PredictionMethod.COSINE:
            U_scaled = _normalize_tensor(U_tensor, axis=1)
            V_scaled = _normalize_tensor(V_tensor, axis=1)
            result = torch.matmul(U_scaled, V_scaled.t())
        else:
            raise ValueError(f"Unsupported prediction method: {method}")

        return result

    U = np.asarray(U, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    if method == PredictionMethod.DOT:
        U_scaled = U
        V_scaled = V
    elif method == PredictionMethod.COSINE:
        U_norm = np.expand_dims(np.linalg.norm(U, ord=2, axis=1), axis=1)
        V_norm = np.expand_dims(np.linalg.norm(V, ord=2, axis=1), axis=1)

        U_norm[U_norm == 0.0] = 1.0
        V_norm[V_norm == 0.0] = 1.0

        U_scaled = np.divide(U, U_norm)
        V_scaled = np.divide(V, V_norm)
    else:
        raise ValueError(f"Unsupported prediction method: {method}")

    return U_scaled @ V_scaled.T
