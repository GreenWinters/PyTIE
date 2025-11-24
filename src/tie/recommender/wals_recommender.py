'''
Wals Recommender

Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender


class WalsRecommender(Recommender):
    """
    A WALS matrix factorization collaborative filtering recommender model.

    Abstraction function:
     AF(U, V) = a matrix factorization collaborative filtering recommendation model
       with user embeddings U and item embeddings V
    """
    def __init__(self, m: int, n: int, k: int = 10, device=None):
        """
        Initializes a new WalsRecommender object.
        Args:
            m: number of entities. Requires m > 0.
            n: number of items. Requires n > 0.
            k: embedding dimension. Requires k > 0.
            device: torch device (cpu or cuda)
        """
        assert m > 0
        assert n > 0
        assert k > 0
        self.device = device if device is not None else torch.device('cpu')
        self._U = torch.zeros((m, k), dtype=torch.float32, device=self.device)
        self._V = torch.zeros((n, k), dtype=torch.float32, device=self.device)
        self._reset_embeddings()
        self._checkrep()

    def _reset_embeddings(self):
        """Resets the embeddings to a standard normal."""
        init_stddev = 1.0
        new_U = torch.normal(mean=0.0, std=init_stddev, size=self._U.shape, device=self.device)
        new_V = torch.normal(mean=0.0, std=init_stddev, size=self._V.shape, device=self.device)
        self._U = new_U
        self._V = new_V

    def _checkrep(self):
        """Asserts the rep invariant."""
        assert self._U is not None
        assert self._V is not None

    @property
    def m(self) -> int:
        """Gets the number of entities represented by the model."""
        self._checkrep()
        return self._U.shape[0]

    @property
    def n(self) -> int:
        """Gets the number of items represented by the model."""
        self._checkrep()
        return self._V.shape[0]

    @property
    def k(self) -> int:
        """Gets the embedding dimension of the model."""
        self._checkrep()
        return self._U.shape[1]

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T. Model must be trained."""
        self._checkrep()
        # Always move to CPU before numpy conversion
        return self._U.detach().cpu().numpy()

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T. Model must be trained."""
        self._checkrep()
        return self._V.detach().cpu().numpy()

    def _update_factor(
        self,
        opposing_factors,
        data,
        alpha: float,
        regularization_coefficient: float
    ) -> torch.Tensor:
        """
        Updates factors according to least squares on the opposing factors (GPU/CPU compatible).
        Args:
            opposing_factors: a pxk tensor of the fixed factors in the optimization step (entity or item factors).
            data: pxq tensor of observed values for each entity/item.
            alpha: Weight for positive training examples (alpha > 0).
            regularization_coefficient: coefficient on the embedding regularization term (>= 0).
        Returns:
            A qxk tensor of recomputed factors which minimize error.
        """
        device = self.device
        V = opposing_factors if torch.is_tensor(opposing_factors) else torch.tensor(opposing_factors, dtype=torch.float32, device=device)
        data = data if torch.is_tensor(data) else torch.tensor(data, dtype=torch.float32, device=device)
        p, k = V.shape
        q = data.shape[1]
        assert p > 0
        assert k == self.k
        assert p == data.shape[0]
        assert q > 0
        assert alpha > 0
        assert regularization_coefficient >= 0

        V_T_V = V.t() @ V  # (k, k)
        new_U = torch.empty((q, k), dtype=torch.float32, device=device)
        P = data
        C = torch.where(P > 0, torch.tensor(alpha + 1, device=device), torch.tensor(1.0, device=device))  # shape (p, q)
        v_outer = torch.einsum('ik,ij->ijk', V, V)  # (p, k, k)
        for i in range(q):
            P_u = P[:, i]
            C_u = C[:, i]
            c_minus_i = C_u - 1
            mask = c_minus_i != 0
            if mask.any():
                confidence_scaled_v_transpose_v = torch.sum(v_outer[mask], dim=0)
            else:
                confidence_scaled_v_transpose_v = torch.zeros((k, k), dtype=torch.float32, device=device)
            try:
                inv = torch.linalg.inv(
                    V_T_V
                    + confidence_scaled_v_transpose_v
                    + regularization_coefficient * torch.eye(k, device=device)
                )
            except RuntimeError as e:
                print(f"[WalsRecommender][ERROR] Matrix inversion failed for column {i}: {e}. Skipping this update.")
                new_U[i, :] = torch.zeros((k,), dtype=torch.float32, device=device)
                continue
            U_i = inv @ V.t() @ P_u
            new_U[i, :] = U_i
        return new_U

    def fit(
        self,
        data,
        epochs: int,
        c: float = 0.024,
        regularization_coefficient: float = 0.01,
        device=None):
        """Fits the model to data (GPU/CPU compatible, with progress and error logging)."""
        if device is not None:
            self.device = device
            self._U = self._U.to(self.device)
            self._V = self._V.to(self.device)
        self._reset_embeddings()

        assert 0 < c < 1

        # Robust input conversion
        if hasattr(data, 'indices') and hasattr(data, 'values') and hasattr(data, 'shape'):
            tensor_data = data
            if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                tensor_data = tensor_data.coalesce()
            indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
            vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
            if torch.is_tensor(indices):
                indices = indices.to(self.device)
            if torch.is_tensor(vals):
                vals = vals.to(self.device)
            # indices shape: (2, nnz) or (nnz, 2)
            if indices.shape[0] == 2:
                row_idx = indices[0]
                col_idx = indices[1]
            elif indices.shape[1] == 2:
                row_idx = indices[:, 0]
                col_idx = indices[:, 1]
            else:
                raise ValueError(f"Unexpected indices shape: {indices.shape}")
            P = torch.zeros((self.m, self.n), dtype=torch.float32, device=self.device)
            P[row_idx, col_idx] = vals
        elif isinstance(data, np.ndarray):
            P = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(data):
            P = data.to(self.device)
        else:
            raise ValueError("Unsupported data format for fit().")
        assert P.shape == (self.m, self.n)
        alpha = (1 / c) - 1
        print(f"[WalsRecommender] Starting training for {epochs} epochs on device {self.device}...")
        for epoch in range(epochs):
            try:
                self._U = self._update_factor(
                    self._V, P.t(), alpha, regularization_coefficient
                )
                self._V = self._update_factor(self._U, P, alpha, regularization_coefficient)
                print(f"[WalsRecommender] Epoch {epoch+1}/{epochs} complete.")
            except Exception as e:
                print(f"[WalsRecommender][ERROR] Training failed at epoch {epoch+1}: {e}")
        print(f"[WalsRecommender] Training finished.")
        self._checkrep()

    def evaluate(
        self,
        test_data,
        method: PredictionMethod = PredictionMethod.DOT) -> float:
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
            values = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
            if torch.is_tensor(indices):
                indices = indices.cpu().numpy()
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            # indices shape: (2, nnz) or (nnz, 2) depending on format
            if indices.shape[0] == 2 and indices.shape[1] == values.shape[0]:
                # (2, nnz) format
                row_indices = indices[0]
                column_indices = indices[1]
            elif indices.shape[1] == 2 and indices.shape[0] == values.shape[0]:
                # (nnz, 2) format
                row_indices = indices[:, 0]
                column_indices = indices[:, 1]
            else:
                raise ValueError(f"Unexpected indices shape: {indices.shape}")
            prediction_values = predictions_matrix[row_indices, column_indices]
            target_values = values
        elif isinstance(test_data, np.ndarray):
            nonzero = test_data.nonzero()
            prediction_values = predictions_matrix[nonzero]
            target_values = test_data[nonzero]
        elif torch.is_tensor(test_data):
            arr = test_data.cpu().numpy()
            nonzero = arr.nonzero()
            prediction_values = predictions_matrix[nonzero]
            target_values = arr[nonzero]
        else:
            raise ValueError("Unsupported test_data format for evaluate().")

        self._checkrep()
        # Ensure both arrays are 1D and have the same length
        if torch.is_tensor(prediction_values):
            prediction_values = prediction_values.detach().cpu().numpy()
        if torch.is_tensor(target_values):
            target_values = target_values.detach().cpu().numpy()
        prediction_values = np.asarray(prediction_values).reshape(-1)
        target_values = np.asarray(target_values).reshape(-1)
        if prediction_values.shape[0] != target_values.shape[0]:
            raise ValueError(f"Inconsistent number of samples for MSE: predictions={prediction_values.shape}, targets={target_values.shape}")
        return mean_squared_error(target_values, prediction_values)

    def predict(self, method: PredictionMethod = PredictionMethod.DOT) -> np.ndarray:
        """Gets the model predictions (always returns CPU numpy array)."""
        self._checkrep()
        pred = calculate_predicted_matrix(self._U, self._V, method)
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        return pred

    def predict_new_entity(
        self,
        entity,
        c: float,
        regularization_coefficient: float,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs,
    ) -> np.array:
        """Recommends items to an unseen entity (GPU/CPU compatible, always returns CPU numpy array)."""
        # Robust input conversion
        if hasattr(entity, 'indices') and hasattr(entity, 'values') and hasattr(entity, 'shape'):
            arr = np.zeros((self.n,))
            tensor_entity = entity
            if hasattr(tensor_entity, 'is_sparse') and tensor_entity.is_sparse:
                tensor_entity = tensor_entity.coalesce()
            indices = tensor_entity.indices() if callable(tensor_entity.indices) else tensor_entity.indices
            values = tensor_entity.values() if callable(tensor_entity.values) else tensor_entity.values
            if torch.is_tensor(indices):
                indices = indices.cpu().numpy()
            if torch.is_tensor(values):
                values = values.cpu().numpy()
            if indices.shape[0] == 2:
                for i in range(indices.shape[1]):
                    arr[indices[1, i]] = values[i]
            elif indices.shape[1] == 2:
                for i in range(indices.shape[0]):
                    arr[indices[i, 1]] = values[i]
            else:
                raise ValueError(f"Unexpected indices shape: {indices.shape}")
            entity = arr
        elif torch.is_tensor(entity):
            entity = entity.cpu().numpy()
        elif isinstance(entity, list):
            entity = np.array(entity)
        if entity.shape != (self.n,):
            raise ValueError(f"Entity shape {entity.shape} does not match expected ({self.n},)")
        alpha = (1 / c) - 1
        new_entity_factor = self._update_factor(
            opposing_factors=self._V,
            data=np.expand_dims(entity, axis=1),
            alpha=alpha,
            regularization_coefficient=regularization_coefficient,
        )
        assert new_entity_factor.shape == (1, self._U.shape[1])
        pred = calculate_predicted_matrix(new_entity_factor, self._V, method)
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        return np.squeeze(pred)

Recommender.register(WalsRecommender)