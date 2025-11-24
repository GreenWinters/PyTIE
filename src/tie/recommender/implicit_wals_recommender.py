'''
Implicit WALs Recommender 


Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import os
import torch
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.metrics import mean_squared_error
from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Fix OpenBLAS threadpool warning for performance


class ImplicitWalsRecommender(Recommender):
    """
    A WALS matrix factorization collaborative filtering recommender model.

    Abstraction function:
    AF(model, m, n, k, num_new_users) = a matrix factorization collaborative filtering
        recommendation model of embedding dimension k with m entity embeddings
        model.user_factors and n item embeddings model.item_factors. The model has
        performed cold start prediction for num_new_users.
    Rep invariant:
        - m > 0
        - n > 0
        - k > 0
    Safety from rep exposure:
        - k is private and immutable
        - model is never returned

    TODO: The implicit library's ALS implementation is CPU-only and does not support GPU acceleration.
    To leverage GPU, consider migrating to a PyTorch or TensorFlow-based ALS implementation (e.g., torch-als, spotlight, or custom PyTorch ALS).
    This will significantly speed up large-scale experiments and remove the current bottleneck.
    """
    def __init__(self, m: int, n: int, k: int = 10, device=None):
        """
        Initializes a new ImplicitWALSRecommender object.

        Args:
            m: number of entities.  Requires m > 0.
            n: number of items.  Requires n > 0.
            k: embedding dimension.  Requires k > 0.
            device: torch device (optional)
        """
        assert m > 0
        assert n > 0
        assert k > 0
        self._m = m
        self._n = n
        self._k = k
        self._model = None
        # for tracking how many new users we've seen so far
        self._num_new_users = 0
        self.device = device if device is not None else torch.device('cpu')
        self._checkrep()

    def to(self, device):
        """
        Moves the model to the specified device for PyTorch compatibility.
        This method does not move internal tensors, but sets the device attribute.
        Args:
            device: torch.device or str, the target device (e.g., 'cuda', 'cpu').
        Returns:
            self
        """
        self.device = device
        return self

    def _log_error(self, context, error):
        """
        Logs errors with context for debugging and experiment robustness.
        Args:
            context: str, description of where the error occurred.
            error: Exception, the error object.
        Returns:
            dict with error details.
        """
        print(f"[ImplicitWalsRecommender][ERROR] Context: {context} | Error: {error}")
        return {"error": str(error), "context": context, "status": "error"}

    def _checkrep(self):
        """
        Asserts the representation invariants for the recommender.
        Ensures m, n, k are positive integers.
        Raises AssertionError if invariants are violated.
        """
        assert self._m > 0
        assert self._n > 0
        assert self._k > 0

    @property
    def U(self) -> np.ndarray:
        """
        Returns the user factor matrix U from the trained model.
        Returns:
            np.ndarray: User factors (shape: m x k)
        Raises:
            AssertionError if model is not trained.
        """
        assert self._model
        self._checkrep()
        return np.copy(self._model.user_factors)

    @property
    def V(self) -> np.ndarray:
        """
        Returns the item factor matrix V from the trained model.
        Returns:
            np.ndarray: Item factors (shape: n x k)
        Raises:
            AssertionError if model is not trained.
        """
        assert self._model
        self._checkrep()
        return np.copy(self._model.item_factors)

    def fit(
        self,
        data,
        epochs: int,
        c: float = 0.024,
        regularization_coefficient: float = 0.01):
        """
        Fits the model to training data using WALS (ALS from implicit library).
        Handles numpy arrays, PyTorch tensors, and sparse formats robustly.
        Catches and logs all errors, never raises.
        Args:
            data: Training data (numpy array, PyTorch tensor, or sparse format).
            epochs: Number of training epochs.
            c: Weight for negative training examples (0 < c < 1).
            regularization_coefficient: Regularization parameter for ALS.
        Returns:
            None or error dict if an error occurs.
        """
        try:
            if not (0 < c < 1):
                raise ValueError(f"Parameter c must be in (0,1), got {c}")
            alpha = (1 / c) - 1
            self._model = AlternatingLeastSquares(
                factors=self._k,
                regularization=regularization_coefficient,
                iterations=epochs,
                alpha=alpha,
            )
            # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
            if hasattr(data, 'indices') and hasattr(data, 'values') and hasattr(data, 'shape'):
                tensor_data = data
                if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                    tensor_data = tensor_data.coalesce()
                indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
                if torch.is_tensor(indices):
                    indices = indices.t().cpu().numpy()
                vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
                if torch.is_tensor(vals):
                    vals = vals.cpu().numpy()
                if indices.shape[0] != vals.shape[0]:
                    raise ValueError(f"Indices and values must have same length. Got {indices.shape[0]} and {vals.shape[0]}")
                if indices.shape[1] != 2:
                    raise ValueError(f"Indices must have shape (nnz, 2). Got {indices.shape}")
                shape = tensor_data.shape if hasattr(tensor_data, 'shape') else (np.max(indices[:,0])+1, np.max(indices[:,1])+1)
                sparse_data = sparse.csr_matrix(
                    (vals, (indices[:, 0], indices[:, 1])), shape=shape
                )
            elif isinstance(data, np.ndarray):
                sparse_data = sparse.csr_matrix(data)
            elif torch.is_tensor(data):
                sparse_data = sparse.csr_matrix(data.cpu().numpy())
            else:
                raise ValueError("Unsupported data format for fit().")
            self._model.fit(sparse_data)
            self._checkrep()
        except Exception as e:
            self._log_error("fit", e)
            # Do not raise, just log and return
            return self._log_error("fit", e)

    def evaluate(
        self,
        test_data,
        method: PredictionMethod = PredictionMethod.DOT) -> float:
        """
        Evaluates the trained model on test data and returns mean squared error.
        Handles numpy arrays, PyTorch tensors, and sparse formats robustly.
        Catches and logs all errors, never raises.
        Args:
            test_data: Test data (numpy array, PyTorch tensor, or sparse format).
            method: Prediction method (dot/cosine).
        Returns:
            float: Mean squared error, or error dict if an error occurs.
        """
        try:
            predictions_matrix = self.predict(method)
            if hasattr(test_data, 'indices') and hasattr(test_data, 'values') and hasattr(test_data, 'shape'):
                tensor_data = test_data
                if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                    tensor_data = tensor_data.coalesce()
                indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
                if torch.is_tensor(indices):
                    indices = indices.t().cpu().numpy()
                vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
                if torch.is_tensor(vals):
                    vals = vals.cpu().numpy()
                if indices.shape[0] != vals.shape[0]:
                    raise ValueError(f"Indices and values must have same length. Got {indices.shape[0]} and {vals.shape[0]}")
                if indices.shape[1] != 2:
                    raise ValueError(f"Indices must have shape (nnz, 2). Got {indices.shape}")
                prediction_values = predictions_matrix[indices[:, 0], indices[:, 1]]
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
            self._checkrep()
            return mean_squared_error(target_values, prediction_values)
        except Exception as e:
            self._log_error("evaluate", e)
            return self._log_error("evaluate", e)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        """
        Returns the full predicted matrix (A_hat) for all users and items.
        Handles model state and device compatibility robustly.
        Catches and logs all errors, never raises.
        Args:
            method: Prediction method (dot/cosine).
        Returns:
            np.ndarray: Predicted matrix, or error dict if an error occurs.
        """
        try:
            self._checkrep()
            return calculate_predicted_matrix(
                self._model.user_factors, self._model.item_factors, method
            )
        except Exception as e:
            self._log_error("predict", e)
            return self._log_error("predict", e)

    def predict_new_entity(
        self,
        entity,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.array:
        """
        Recommends items to an unseen entity (cold-start prediction).
        Handles numpy arrays, PyTorch tensors, and sparse formats robustly.
        Catches and logs all errors, never raises.
        Args:
            entity: Ratings for the new entity (numpy array, PyTorch tensor, or sparse format).
            method: Prediction method (dot/cosine).
        Returns:
            np.ndarray: Predicted scores for the new entity, or error dict if an error occurs.
        """
        try:
            # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
            if hasattr(entity, 'indices') and hasattr(entity, 'values') and hasattr(entity, 'shape'):
                tensor_data = entity
                if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                    tensor_data = tensor_data.coalesce()
                indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
                values = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
                if torch.is_tensor(indices):
                    indices = indices.cpu().numpy()
                if torch.is_tensor(values):
                    values = values.cpu().numpy()
                # Auto-correct common mistake: shape (2, nnz) instead of (nnz, 2)
                if indices.ndim == 2 and indices.shape[0] == 2 and indices.shape[1] != 2:
                    print(f"[ImplicitWalsRecommender][WARN] Auto-transposing entity indices from shape {indices.shape} to ({indices.shape[1]}, 2)")
                    indices = indices.T
                # indices shape: (nnz, 2), values shape: (nnz,)
                if indices.ndim != 2 or indices.shape[1] != 2:
                    raise ValueError(f"Entity indices must have shape (nnz, 2). Got {indices.shape}")
                if len(values) != indices.shape[0]:
                    raise ValueError(f"Entity values and indices must have same length. Got {len(values)} and {indices.shape[0]}")
                row_indices = np.zeros(len(values), dtype=int)
                column_indices = indices[:, 1]
                # Robustly determine shape
                if hasattr(tensor_data, 'shape') and len(tensor_data.shape) > 1:
                    n_items = tensor_data.shape[1]
                elif hasattr(tensor_data, 'shape'):
                    n_items = tensor_data.shape[0]
                else:
                    n_items = max(column_indices) + 1 if len(column_indices) > 0 else len(values)
                sparse_data = sparse.csr_matrix(
                    (values, (row_indices, column_indices)), shape=(1, n_items)
                )
            elif isinstance(entity, np.ndarray):
                if entity.ndim == 1:
                    sparse_data = sparse.csr_matrix(entity.reshape(1, -1))
                elif entity.ndim == 2 and entity.shape[0] == 1:
                    sparse_data = sparse.csr_matrix(entity)
                else:
                    raise ValueError(f"Entity ndarray must be 1D or 2D with shape (1, n). Got {entity.shape}")
            elif torch.is_tensor(entity):
                arr = entity.cpu().numpy()
                if arr.ndim == 1:
                    sparse_data = sparse.csr_matrix(arr.reshape(1, -1))
                elif arr.ndim == 2 and arr.shape[0] == 1:
                    sparse_data = sparse.csr_matrix(arr)
                else:
                    raise ValueError(f"Entity tensor must be 1D or 2D with shape (1, n). Got {arr.shape}")
            else:
                raise ValueError("Unsupported entity format for predict_new_entity().")

            user_id = self._m + self._num_new_users
            self._model.partial_fit_users((user_id,), sparse_data)
            self._num_new_users += 1
            self._checkrep()
            return np.squeeze(
                calculate_predicted_matrix(
                    np.expand_dims(self._model.user_factors[user_id], axis=1).T,
                    self._model.item_factors,
                    method,
                )
            )
        except Exception as e:
            self._log_error("predict_new_entity", e)
            return self._log_error("predict_new_entity", e)
