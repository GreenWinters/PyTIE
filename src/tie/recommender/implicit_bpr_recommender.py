'''
Implicit BPR Recommender 


Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import numpy as np
import torch
from implicit.bpr import BayesianPersonalizedRanking
from scipy import sparse
from sklearn.metrics import mean_squared_error

from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix


class ImplicitBPRRecommender:
    """
    A matrix factorization recommender model to suggest items for an entity.

    Abstraction function:
       AF(m, n, k, model, num_new_users) = model to be trained with embedding dimension
       k
           on m entities and n items if model is None,
           or model trained on such data and with predictions for num_new_users
           if model is not None
    
    Rep invariant:
       - m > 0
       - n > 0
       - k > 0
       - num_new_users >= 0
    
    Safety from rep exposure:
       - k is private and immutable
       - model is never returned
    """
    def __init__(self, m: int, n: int, k: int, device=None):
        """Initializes an ImplicitBPRRecommender object.

        Args:
            m: number of entities.  Requires m > 0.
            n: number of items.  Requires n > 0.
            k: the embedding dimension.  Requires k > 0.
            device: torch device (optional)
        """
        # assert preconditions
        assert k > 0
        assert n > 0
        assert k > 0

        self._m = m
        self._n = n
        self._k = k
        self._model = None

        self._num_new_users = 0
        self.device = device if device is not None else torch.device('cpu')

        self._checkrep()

    def to(self, device):
        """Moves model to the specified device (for PyTorch compatibility)."""
        self.device = device
        # No internal tensors to move, but method provided for compatibility
        return self

    def _checkrep(self):
        """Asserts the rep invariant."""
        #   - m > 0
        assert self._m > 0
        #   - n > 0
        assert self._n > 0
        #   - k > 0
        assert self._k > 0
        #   - num_new_users >= 0
        assert self._num_new_users >= 0

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T. Model must be trained."""
        assert self._model

        self._checkrep()
        return np.copy(self._model.user_factors)

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T. Model must be trained."""
        assert self._model

        self._checkrep()
        return np.copy(self._model.item_factors)

    def fit(
        self,
        data,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        device=None,
        **kwargs,
    ):
        """Fits the model to data (with device and tensor handling)."""
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')

        self._model = BayesianPersonalizedRanking(
            factors=self._k,
            learning_rate=learning_rate,
            regularization=regularization_coefficient,
            iterations=epochs,
            verify_negative_samples=True,
        )

        # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
        if hasattr(data, 'indices') and hasattr(data, 'values') and hasattr(data, 'shape'):
            # PyTorch sparse tensor handling
            tensor_data = data
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
            sparse_data = sparse.csr_matrix(
                (vals, (row_indices, column_indices)), shape=tensor_data.shape
            )
        elif isinstance(data, np.ndarray):
            sparse_data = sparse.csr_matrix(data)
        elif torch.is_tensor(data):
            sparse_data = sparse.csr_matrix(data.cpu().numpy())
        else:
            raise ValueError("Unsupported data format for fit().")

        self._model.fit(sparse_data)
        self._checkrep()

    def evaluate(
        self,
        test_data,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> float:
        """Evaluates the solution.

        Requires that the model has been trained.

        Args:
            test_data: mxn tensor on which to evaluate the model.
                Requires that mxn match the dimensions of the training tensor and
                each row i and column j correspond to the same entity and item
                in the training tensor, respectively.
            method: The prediction method to use.

        Returns:
            The mean squared error of the test data.
        """
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

        self._checkrep()
        return mean_squared_error(target_values, prediction_values)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        """Gets the model predictions.

        The predictions consist of the estimated matrix A_hat of the truth
        matrix A, of which the training data contains a sparse subset of the entries.

        Args:
            method: The prediction method to use.

        Returns:
            An mxn array of values.
        """
        self._checkrep()

        return calculate_predicted_matrix(
            self._model.user_factors, self._model.item_factors, method
        )

    def predict_new_entity(
        self,
        entity,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.array:
        # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
        if torch.is_tensor(entity):
            if entity.is_sparse:
                entity = entity.to_dense()
            entity = entity.cpu().numpy()
        elif hasattr(entity, 'toarray'):
            entity = entity.toarray()
        elif isinstance(entity, np.ndarray):
            pass
        else:
            raise ValueError("Unsupported entity format for predict_new_entity().")

        # Use average user factor for cold-start prediction
        avg_user_factor = np.mean(self._model.user_factors, axis=0)
        scores = np.dot(avg_user_factor, self._model.item_factors.T)
        return scores
