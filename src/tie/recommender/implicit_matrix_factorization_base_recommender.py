"""Wrapper around implicit.gpu.matrix_factorization_base.MatrixFactorizationBase."""

from typing import Optional, Type

import numpy as np
from scipy import sparse
import torch
from implicit.cpu.als import AlternatingLeastSquares as CpuAlternatingLeastSquares
from implicit.gpu.als import AlternatingLeastSquares as GpuAlternatingLeastSquares
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase as CpuMatrixFactorizationBase
from implicit.gpu.matrix_factorization_base import MatrixFactorizationBase as GpuMatrixFactorizationBase

from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender
from sklearn.metrics import mean_squared_error


def _ensure_csr(data):
    if hasattr(data, 'indices') and hasattr(data, 'values') and hasattr(data, 'shape'):
        tensor_data = data
        if hasattr(tensor_data, 'coalesce'):
            tensor_data = tensor_data.coalesce()
        indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
        values = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
        if torch.is_tensor(indices):
            indices = indices.t().cpu().numpy()
        if torch.is_tensor(values):
            values = values.cpu().numpy()
        shape = tensor_data.shape
        return sparse.csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)
    if isinstance(data, np.ndarray):
        return sparse.csr_matrix(data)
    if torch.is_tensor(data):
        return sparse.csr_matrix(data.cpu().numpy())
    raise ValueError("Unsupported data format for MatrixFactorizationBase")


class ImplicitMatrixFactorizationBaseRecommender(Recommender):
    """Adapter that exposes MatrixFactorizationBase inference helpers."""

    def __init__(
        self,
        m: int,
        n: int,
        base_model: Optional[Type[GpuMatrixFactorizationBase | CpuMatrixFactorizationBase]] = None,
        **model_kwargs,
    ):
        assert m > 0 and n > 0
        self._gpu_base_cls = base_model or GpuAlternatingLeastSquares
        self._cpu_base_cls = CpuAlternatingLeastSquares
        self._model_kwargs = model_kwargs
        self._model: Optional[GpuMatrixFactorizationBase | CpuMatrixFactorizationBase] = None
        self._m = m
        self._n = n
        self._using_gpu = True

    def fit(self, data, **kwargs):
        csr = _ensure_csr(data)
        try:
            self._model = self._create_model(use_gpu=self._using_gpu)
            self._model.fit(csr)
        except RuntimeError as exc:
            if self._using_gpu:
                print(
                    "[ImplicitMatrixFactorizationBase] GPU training failed ({exc}). Falling back to CPU implementation.".format(
                        exc=exc
                    )
                )
                self._using_gpu = False
                self._model = self._create_model(use_gpu=False)
                self._model.fit(csr)
            else:
                raise

    def _create_model(self, use_gpu: bool) -> GpuMatrixFactorizationBase | CpuMatrixFactorizationBase:
        cls = self._gpu_base_cls if use_gpu else self._cpu_base_cls
        return cls(**self._model_kwargs)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        assert self._model is not None
        return calculate_predicted_matrix(
            self._resolve_factors(self._model.user_factors), self._resolve_factors(self._model.item_factors), method
        )

    def evaluate(
        self,
        test_data,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> float:
        predictions_matrix = self.predict(method)

        if hasattr(test_data, 'indices') and hasattr(test_data, 'values') and hasattr(test_data, 'shape'):
            tensor_data = test_data
            if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                tensor_data = tensor_data.coalesce()
            indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
            row_indices = tuple(indices[:, 0])
            column_indices = tuple(indices[:, 1])
            vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
            if torch.is_tensor(vals):
                vals = vals.cpu().numpy()
            prediction_values = predictions_matrix[row_indices, column_indices]
            target_values = vals
        elif isinstance(test_data, np.ndarray):
            prediction_values = predictions_matrix[test_data.nonzero()]
            target_values = test_data[test_data.nonzero()]
        elif torch.is_tensor(test_data):
            arr = test_data.cpu().numpy()
            prediction_values = predictions_matrix[arr.nonzero()]
            target_values = arr[arr.nonzero()]
        else:
            raise ValueError("Unsupported test_data format for evaluate().")

        return mean_squared_error(target_values, prediction_values)

    def recommend(self, user_id, user_items, **kwargs):
        assert self._model is not None
        return self._model.recommend(user_id, user_items, **kwargs)

    def similar_users(self, user_id, **kwargs):
        assert self._model is not None
        return self._model.similar_users(user_id, **kwargs)

    def similar_items(self, item_id, **kwargs):
        assert self._model is not None
        return self._model.similar_items(item_id, **kwargs)

    @property
    def U(self):
        assert self._model is not None
        return np.copy(self._resolve_factors(self._model.user_factors))

    @property
    def V(self):
        assert self._model is not None
        return np.copy(self._resolve_factors(self._model.item_factors))

    def predict_new_entity(
        self,
        entity,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.ndarray:
        assert self._model is not None
        entity_vector = self._prepare_entity_vector(entity)
        item_factors = self._resolve_factors(self._model.item_factors)
        if entity_vector.shape[-1] != item_factors.shape[0]:
            raise ValueError("Entity vector length must match the number of items.")
        user_factor = entity_vector @ item_factors
        predictions = calculate_predicted_matrix(
            user_factor[np.newaxis, :], item_factors, method
        )
        return np.asarray(predictions).reshape(-1)

    def _resolve_factors(self, values):
        if hasattr(values, 'data'):
            return values.data
        return values

    def _prepare_entity_vector(self, entity) -> np.ndarray:
        if torch.is_tensor(entity):
            tensor = entity
            if tensor.is_sparse:
                tensor = tensor.to_dense()
            vector = tensor.cpu().numpy()
        elif hasattr(entity, 'toarray'):
            vector = entity.toarray()
        elif isinstance(entity, np.ndarray):
            vector = entity
        else:
            raise ValueError("Unsupported entity format for predict_new_entity().")

        vector = np.asarray(vector)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        if vector.ndim != 1:
            raise ValueError("Entity must be a 1-dimensional vector.")
        return vector
