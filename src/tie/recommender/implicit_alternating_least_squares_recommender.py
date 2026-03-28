"""
Implicit Alternating Least Squares Recommender

Adapted for the implicit GPU implementation of ALS; keeps the same Recommender
interface used throughout the Technique Inference Engine. The GPU backend
handles sparse data efficiently while still exposing the usual user/item factor
matrices and prediction helpers.
"""
import os
from typing import Optional

import numpy as np
from scipy import sparse
import torch
from implicit.cpu.als import AlternatingLeastSquares as CpuAlternatingLeastSquares
from implicit.gpu.als import AlternatingLeastSquares as GpuAlternatingLeastSquares

from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender
from sklearn.metrics import mean_squared_error

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def _to_csr_matrix(data):
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
    raise ValueError("Unsupported data format for ALS fit()")


class ImplicitAlternatingLeastSquaresRecommender(Recommender):
    """GPU-backed ALS recommender."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int = 64,
        regularization: float = 0.01,
        alpha: float = 1.0,
        iterations: int = 15,
        calculate_training_loss: bool = False,
        random_state: Optional[int] = None,
    ):
        assert m > 0 and n > 0 and k > 0
        self._m = m
        self._n = n
        self._k = k
        self._regularization = regularization
        self._alpha = alpha
        self._iterations = iterations
        self._calculate_training_loss = calculate_training_loss
        self._random_state = random_state
        self._model_cls = GpuAlternatingLeastSquares
        self._cpu_model_cls = CpuAlternatingLeastSquares
        self._using_gpu = True
        try:
            self._model = self._create_model(use_gpu=True)
        except RuntimeError as exc:
            print(
                "[ImplicitALS] GPU initialization failed ({exc}). Falling back to CPU implementation.".format(
                    exc=exc
                )
            )
            self._using_gpu = False
            self._model = self._create_model(use_gpu=False)

    def _create_model(self, use_gpu: bool):
        cls = self._model_cls if use_gpu else self._cpu_model_cls
        return cls(
            factors=self._k,
            regularization=self._regularization,
            alpha=self._alpha,
            iterations=self._iterations,
            calculate_training_loss=self._calculate_training_loss,
            random_state=self._random_state,
        )

    @property
    def U(self) -> np.ndarray:
        assert self._model is not None
        return np.copy(self._resolve_factors(self._model.user_factors))

    @property
    def V(self) -> np.ndarray:
        assert self._model is not None
        return np.copy(self._resolve_factors(self._model.item_factors))

    def _resolve_factors(self, values):
        if hasattr(values, 'data'):
            return values.data
        return values

    def fit(
        self,
        data,
        **kwargs,
    ):
        csr = _to_csr_matrix(data)
        try:
            self._model.fit(csr)
        except RuntimeError as exc:
            if self._using_gpu:
                print(
                    "[ImplicitALS] GPU training failed ({exc}). Falling back to CPU implementation.".format(
                        exc=exc
                    )
                )
                self._using_gpu = False
                self._model = self._create_model(use_gpu=False)
                self._model.fit(csr)
            else:
                raise

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

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        assert self._model is not None
        return calculate_predicted_matrix(
            self._resolve_factors(self._model.user_factors), self._resolve_factors(self._model.item_factors), method
        )

    def predict_new_entity(self, entity, method: PredictionMethod = PredictionMethod.DOT):
        if hasattr(entity, 'toarray'):
            entity = entity.toarray()
        if hasattr(entity, 'numpy'):
            entity = entity.numpy()
        return self._model.recommend(0, sparse.csr_matrix(entity), N=entity.shape[1])
