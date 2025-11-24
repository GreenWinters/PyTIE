'''
Factorization Recommender 

Code adapted from https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb?utm_source=ss-recommendation-systems&utm_campaign=colab-external&utm_medium=referral&utm_content=recommendation-systems

Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import copy
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender


class FactorizationRecommender(Recommender):
    """A matrix factorization collaborative filtering recommender model."""

    # Abstraction function:
    #   AF(m, n, k) = a matrix factorization recommender model
    #       on m entities, n items to recommend, and
    #       embedding dimension k (a hyperparameter)
    # Rep invariant:
    #   - U.shape[1] == V.shape[1]
    #   - U and V are 2D
    #   - U.shape[0] > 0
    #   - U.shape[1] > 0
    #   - V.shape[0] > 0
    #   - V.shape[1] > 0
    #   - all elements of U are non-null
    #   - all elements of V are non-null
    #   - loss is not None
    # Safety from rep exposure:
    #   - U and V are private and not reassigned
    #   - methods to get U and V return a deepcopy of the numpy representation
    def __init__(self, m, n, k, device=None):
        """Initializes a FactorizationRecommender object.

        Args:
            m: number of entities
            n: number of items
            k: embedding dimension
            device: torch device (cpu or cuda)
        """
        self.device = device if device is not None else torch.device('cpu')
        self._U = torch.zeros((m, k), dtype=torch.float32, device=self.device)
        self._V = torch.zeros((n, k), dtype=torch.float32, device=self.device)
        self._reset_embeddings()
        self._loss = torch.nn.MSELoss().to(self.device)
        self._init_stddev = 1
        self._checkrep()

    def to(self, device):
        self.device = device
        self._U = self._U.to(device)
        self._V = self._V.to(device)
        self._loss = self._loss.to(device)
        return self

    def _reset_embeddings(self):
        """Resets the embeddings to a standard normal."""
        init_stddev = 1
        self._U = torch.normal(mean=0, std=init_stddev, size=self._U.shape, device=self.device)
        self._V = torch.normal(mean=0, std=init_stddev, size=self._V.shape, device=self.device)

    def _checkrep(self):
        """Asserts the rep invariant."""
        #   - U.shape[1] == V.shape[1]
        assert self._U.shape[1] == self._V.shape[1]
        #   - U and V are 2D
        assert len(self._U.shape) == 2
        assert len(self._V.shape) == 2
        #   - U.shape[0] > 0
        assert self._U.shape[0] > 0
        #   - U.shape[1] > 0
        assert self._U.shape[1] > 0
        #   - V.shape[0] > 0
        assert self._V.shape[0] > 0
        #   - V.shape[1] > 0
        assert self._V.shape[1] > 0
        #   - all elements of U are non-null
        assert not torch.isnan(self._U).any()
        #   - all elements of V are non-null
        assert not torch.isnan(self._V).any()
        #   - loss is not None
        assert self._loss is not None

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T."""
        self._checkrep()
        return copy.deepcopy(self._U.cpu().numpy())

    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T."""
        self._checkrep()
        return copy.deepcopy(self._V.cpu().numpy())

    def _get_estimated_matrix(self) -> torch.Tensor:
        """Gets the estimated matrix UV^T."""
        self._checkrep()
        return torch.matmul(self._U, self._V.t())

    def _predict(self, data) -> torch.Tensor:
        """Predicts the results for data.

        Requires that data be the same shape as the training data.
        Where each row corresponds to the same entity as the training data
        and each column represents the same item to recommend.  However,
        the tensor may be sparse and contain more, fewer, or the same number
        of entries as the training data.

        Args:
            data: An mxn sparse tensor of data containing p nonzero entries.

        Returns:
            A length-p tensor of predictions, where predictions[i] corresponds to the
                prediction for index data.indices[i].
        """
        self._checkrep()
        est_matrix = self._get_estimated_matrix()
        indices = None
        if hasattr(data, 'is_sparse') and getattr(data, 'is_sparse', False):
            data = data.coalesce()
            indices = data.indices() if callable(data.indices) else data.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
        elif hasattr(data, 'indices'):
            indices = data.indices() if callable(data.indices) else data.indices
        elif isinstance(data, np.ndarray):
            # Dense numpy array: use nonzero indices
            indices = np.argwhere(data != 0)
        else:
            indices = np.array(data)
        if indices is None or indices.size == 0:
            return torch.tensor([])
        if indices.ndim == 1:
            indices = indices.reshape(-1, 2)
        return est_matrix[indices[:, 0], indices[:, 1]]

    def _calculate_regularized_loss(
        self,
        data,
        predictions,
        regularization_coefficient: float,
        gravity_coefficient: float,
    ) -> float:
        r"""Gets the regularized loss function.

        The regularized loss is the sum of:
        - The MSE between data and predictions.
        - A regularization term which is the average of the squared norm of each
            entity embedding, plus the average of the squared norm of each item
            embedding r = 1/m \sum_i ||U_i||^2 + 1/n \sum_j ||V_j||^2
        - A gravity term which is the average of the squares of all predictions.
            g = 1/(MN) \sum_{ij} (UV^T)_{ij}^2

        Args:
            data: the data on which to evaluate.  Predictions will be evaluated for
                every non-null entry of data.
            predictions: the model predictions on which to evaluate.  Requires that
                predictions[i] contains the predictions for data.indices[i].
            regularization_coefficient: the coefficient for the regularization component
                of the loss function.
            gravity_coefficient: the coefficient for the gravity component of the loss
                function.

        Returns:
            The regularized loss.
        """
        regularization_loss = regularization_coefficient * (
            torch.sum(self._U * self._U) / self._U.shape[0]
            + torch.sum(self._V * self._V) / self._V.shape[0]
        )

        gravity = (1.0 / (self._U.shape[0] * self._V.shape[0])) * torch.sum(
            torch.square(torch.matmul(self._U, self._V.t()))
        )

        gravity_loss = gravity_coefficient * gravity

        self._checkrep()
        return self._loss(data, predictions) + regularization_loss + gravity_loss

    def _calculate_mean_square_error(self, data) -> torch.Tensor:
        """Calculates the mean squared error between observed values in the
        data and predictions from UV^T.

        MSE = \sum_{(i,j) \in \Omega} (data_{ij} U_i \dot V_j)^2
        where Omega is the set of observed entries in training_data.

        Args:
            data: A matrix of observations of dense_shape m, n

        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = self._predict(data)
        loss = self._loss(torch.tensor(data.values, dtype=torch.float32), predictions)
        self._checkrep()

    def fit(
        self,
        data,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float = 0.1,
        gravity_coefficient: float = 0.0,
        device=None,
        ):
        """
        Train the factorization recommender on the provided interaction data using
        stochastic gradient descent (SGD).

        This method converts the supplied data into a tensor of observed interaction
        values, computes predictions via the internal _predict routine, and optimizes
        U and V latent factor matrices with a regularized loss (and optional 'gravity'
        term) over (epochs + 1) iterations. After optimization the latent factor
        parameters are detached from the computational graph and an internal
        representation check (_checkrep) is performed.

        Parameters
        ----------
        data : torch.Tensor | numpy.ndarray | Any
            Interaction/source data. Acceptable forms:
              - A PyTorch sparse tensor (with is_sparse attribute or is_sparse True),
                in which case its non-zero values are used.
              - Any object exposing a .values attribute or callable returning values.
              - A NumPy ndarray (non-zero entries are extracted).
              - Any iterable convertible to a NumPy array.
            Only the non-zero / provided values are used to compute the loss.
        learning_rate : float
            Learning rate passed to torch.optim.SGD.
        epochs : int
            Number of training epochs. Note: the loop runs (epochs + 1) times,
            resulting in one extra optimization step (potential off-by-one).
        regularization_coefficient : float, default=0.1
            Coefficient for L2 (weight decay) style regularization applied within
            _calculate_regularized_loss to latent factors U and V.
        gravity_coefficient : float, default=0.0
            Strength of an optional "gravity" term encouraging a structural property
            in the latent space (exact behavior depends on _calculate_regularized_loss).
        device : torch.device | str | None, optional
            Target device ('cpu', 'cuda', etc.). If provided, U and V are moved
            before training. If None, the existing self.device is used.
        gravity_coefficient : float, default=0.0
            (Duplicate parameter name in signature.) This second occurrence overrides
            the first default. This is likely an implementation bug—only the latter
            value is actually bound. Remove duplication to avoid confusion.

        Returns
        -------
        None
            The method updates internal state (self._U, self._V) in place.

        Side Effects
        ------------
        - Re-wraps self._U and self._V as torch.nn.Parameter for optimization.
        - Performs in-place SGD updates.
        - Detaches self._U and self._V after training (no further gradients).
        - Calls self._checkrep() for internal validation.

        Notes
        -----
        - If data is a NumPy array, only entries where data != 0 are included;
          zero entries are ignored (implicit feedback assumption).
        - If data has a .values attribute (callable or property), those are used
          directly without filtering zeros unless the source itself excludes them.
        - Loss computation relies on _calculate_regularized_loss; ensure its
          interface matches the (vals, predictions, regularization_coefficient,
          gravity_coefficient) call pattern.
        - Duplicate gravity_coefficient in the function signature should be fixed.
          Retain only one parameter to prevent shadowing and confusion.

        Potential Improvements
        ----------------------
        - Resolve duplicate gravity_coefficient parameter.
        - Clarify expected shape alignment between vals and predictions.
        - Consider removing the (epochs + 1) extra iteration or document rationale.
        - Support passing sample weights or masking explicitly instead of
          inferring from non-zero values only.
        """
        if device is not None:
            self.device = device
        self._U = torch.nn.Parameter(self._U.to(self.device))
        self._V = torch.nn.Parameter(self._V.to(self.device))
        optimizer = torch.optim.SGD([self._U, self._V], lr=learning_rate)

        for _ in range(epochs + 1):
            optimizer.zero_grad()
            predictions = self._predict(data)
            vals = None
            if hasattr(data, 'is_sparse') and getattr(data, 'is_sparse', False):
                data = data.coalesce()
                vals = data.values() if callable(data.values) else data.values
                if torch.is_tensor(vals):
                    vals = vals.to(self.device)
            elif hasattr(data, 'values'):
                if callable(data.values):
                    vals = data.values()  
                else:
                    vals = data.values
                if torch.is_tensor(vals):
                    vals = vals.to(self.device)
            elif isinstance(data, np.ndarray):
                vals = torch.tensor(data[data != 0], dtype=torch.float32, device=self.device)
            else:
                vals = torch.tensor(np.array(data), dtype=torch.float32, device=self.device)
            loss = self._calculate_regularized_loss(
                vals,
                predictions,
                regularization_coefficient,
                gravity_coefficient,
            )
            loss.backward()
            optimizer.step()
        # Detach parameters after training
        self._U = self._U.detach()
        self._V = self._V.detach()
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

        indices = None
        vals = None
        if hasattr(test_data, 'is_sparse') and getattr(test_data, 'is_sparse', False):
            test_data = test_data.coalesce()
            indices = test_data.indices() if callable(test_data.indices) else test_data.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
            vals = test_data.values() if callable(test_data.values) else test_data.values
            if torch.is_tensor(vals):
                vals = vals.cpu().numpy()
        elif hasattr(test_data, 'indices'):
            indices = test_data.indices() if callable(test_data.indices) else test_data.indices
            vals = test_data.values() if callable(test_data.values) else test_data.values
        elif isinstance(test_data, np.ndarray):
            indices = np.argwhere(test_data != 0)
            vals = test_data[test_data != 0]
        else:
            indices = np.array(test_data)
            vals = np.array(test_data)
        if indices is None or indices.size == 0:
            return 0.0
        if indices.ndim == 1:
            indices = indices.reshape(-1, 2)
        row_indices = tuple(indices[:, 0])
        column_indices = tuple(indices[:, 1])
        prediction_values = predictions_matrix[row_indices, column_indices]
        self._checkrep()
        return mean_squared_error(vals, prediction_values)


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
            np.nan_to_num(self._U.cpu().numpy()), np.nan_to_num(self._V.cpu().numpy()), method
        )

    def predict_new_entity(
        self,
        entity,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        gravity_coefficient: float,
        method: PredictionMethod = PredictionMethod.DOT,
        ) -> np.array:
        """Recommends items to an unseen entity

        Args:
            entity: a length-n sparse tensor of consisting of the new entity's
                ratings for each item, indexed exactly as the items used to
                train this model.
            learning rate: the learning rate for SGD.
            epochs: Number of training epochs, where each the model is trained on the
                cardinality dataset in each epoch.
            regularization_coefficient: coefficient on the embedding regularization
                term.
            gravity_coefficient: coefficient on the prediction regularization term.
            method: The prediction method to use.

        Returns:
            An array of predicted values for the new entity.
        """
        # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
        if hasattr(entity, 'is_sparse') and getattr(entity, 'is_sparse', False):
            entity = entity.coalesce()
            indices = entity.indices() if callable(entity.indices) else entity.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
            vals = entity.values() if callable(entity.values) else entity.values
            if torch.is_tensor(vals):
                vals = vals.cpu().numpy()
            arr = np.zeros((self._V.shape[0],))
            for idx, val in zip(indices, vals):
                arr[idx[0]] = val
            entity = arr
        elif hasattr(entity, 'indices') and hasattr(entity, 'values') and hasattr(entity, 'shape'):
            indices = entity.indices if not callable(entity.indices) else entity.indices()
            vals = entity.values if not callable(entity.values) else entity.values()
            arr = np.zeros((self._V.shape[0],))
            for idx, val in zip(indices, vals):
                arr[idx[0]] = val
            entity = arr
        elif torch.is_tensor(entity):
            entity = entity.cpu().numpy()
        elif isinstance(entity, np.ndarray):
            # Already dense
            pass
        else:
            entity = np.array(entity)
        embedding = torch.nn.Parameter(torch.normal(mean=0, std=self._init_stddev, size=(self._U.shape[1], 1)))
        optimizer = torch.optim.SGD([embedding], lr=learning_rate)
        V_tensor = self._V
        embedding = embedding.to(V_tensor.device)
        entity_indices = None
        entity_values = None
        if isinstance(entity, np.ndarray):
            entity_indices = np.arange(len(entity))
            entity_values = torch.tensor(entity, dtype=torch.float32, device=V_tensor.device)
        else:
            # fallback for other types
            entity_indices = np.arange(len(entity))
            entity_values = torch.tensor(entity, dtype=torch.float32, device=V_tensor.device)

        for _ in range(epochs + 1):
            optimizer.zero_grad()
            predictions = torch.matmul(V_tensor, embedding).squeeze()
            if hasattr(entity, 'indices') and hasattr(entity, 'values'):
                pred_values = predictions[entity_indices[:, 0]]
            else:
                pred_values = predictions[entity_indices]
            mse_loss = self._loss(pred_values, entity_values)
            reg_loss = regularization_coefficient * torch.sum(embedding ** 2) / self._U.shape[0]
            gravity_loss = (gravity_coefficient / (self._U.shape[0] * self._V.shape[0])) * torch.sum(torch.square(torch.matmul(V_tensor, embedding)))
            loss = mse_loss + reg_loss + gravity_loss
            loss.backward()
            optimizer.step()
        emb_np = embedding.detach().cpu().numpy()
        assert not np.isnan(emb_np).any()
        self._checkrep()
        return np.squeeze(
            calculate_predicted_matrix(emb_np.T, self._V.cpu().numpy(), method)
        )


Recommender.register(FactorizationRecommender)
