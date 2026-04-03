'''
BPR Recommender 


Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import math
import numpy as np
import torch
import time
from ..constants import PredictionMethod
from ..utils import calculate_predicted_matrix
from .recommender import Recommender


class BPRRecommender(Recommender):
    """
    A Bayesian Personalized Ranking recommender.

    Abstraction function:
     	AF(U, V) = a Bayesian Personalized Ranking recommender model
           on entity embeddings U and item embeddings V
    
    Rep invariant:
       - U.shape[1] == V.shape[1]
       - U and V are 2D
       - U.shape[0] > 0
       - U.shape[1] > 0
       - V.shape[0] > 0
       - V.shape[1] > 0
    
    Safety from rep exposure:
    Based on BPR: Bayesian Personalized Ranking from Implicit Feedback.
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
    """
    def __init__(self, m: int, n: int, k: int, device=None):
        """
        Initializes a BPRRecommender object.

        Args:
            m: number of entity embeddings.
            n: number of item embeddings.
            k: embedding dimension.
            device: torch device (cpu or cuda)
        """
        self.device = device if device is not None else torch.device('cpu')
        self._U = torch.zeros((m, k), dtype=torch.float32, device=self.device)
        self._V = torch.zeros((n, k), dtype=torch.float32, device=self.device)
        self._reset_embeddings()
        self._checkrep()


    def to(self, device):
        self.device = device
        self._U = self._U.to(device)
        self._V = self._V.to(device)
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

    @property
    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T."""
        return self._U.cpu().numpy()


    @property
    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T."""
        return self._V.cpu().numpy()


    def _sample_dataset(
        self,
        data,
        num_samples: int,
    ) -> tuple:
        """
        Samples the dataset according to the bootstrapped sampling for BPR (PyTorch/GPU compatible).
        
        Sampling is performed uniformly over all triples of the form (u, i, j),
        where u is a user, i is an item for which there is an observation for that user,
        and j is an item for which there is no observation for that user.

        Args:
            data: An mxn matrix of observations.
            num_samples: Number of samples to draw. Requires num_samples > 0.

        Returns:
            A tuple of the form (u, i, j) where u is an array of user indices,
            i is an array of item indices with an observation for that user,
            and j is an array of item indices with no observation for that user.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        if isinstance(data, np.ndarray):
            tensor_data = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(data):
            tensor_data = data.to(dtype=torch.float32, device=self.device)
        else:
            tensor_data = torch.tensor(np.array(data), dtype=torch.float32, device=self.device)

        m, n = tensor_data.shape
        sample_user_probability = self._calculate_sample_user_probability(tensor_data)
        num_items_per_user = torch.sum(tensor_data, dim=1).float()
        num_items_per_user = torch.where(
            num_items_per_user == 0,
            torch.full_like(num_items_per_user, float("nan")),
            num_items_per_user,
        )
        sample_item_probability = torch.nan_to_num(
            tensor_data / num_items_per_user.unsqueeze(1)
        )

        joint_user_item_probability = (
            sample_user_probability.unsqueeze(1) * sample_item_probability
        )
        flattened_probability = joint_user_item_probability.reshape(-1)
        total_probability = flattened_probability.sum()
        if total_probability == 0:
            flattened_probability = torch.full_like(flattened_probability, 1.0 / flattened_probability.numel())
        else:
            flattened_probability = flattened_probability / total_probability

        sampled = torch.multinomial(flattened_probability, num_samples=num_samples, replacement=True)
        all_u = (sampled // n).to(torch.long)
        all_i = (sampled % n).to(torch.long)

        non_observations = (tensor_data == 0).to(torch.float32)
        all_j = torch.empty_like(all_u)
        unique_users, counts = torch.unique(all_u, return_counts=True)
        for user, count in zip(unique_users.tolist(), counts.tolist()):
            row = non_observations[user]
            row_sum = row.sum()
            if row_sum == 0:
                choices = torch.randint(0, n, size=(count,), device=self.device)
            else:
                probs = row / row_sum
                choices = torch.multinomial(probs, num_samples=count, replacement=True)
            positions = (all_u == user).nonzero(as_tuple=True)[0]
            all_j[positions] = choices

        return all_u, all_i, all_j



    def fit(
        self,
        data,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        device=None,
        debug=False):
        """
        Fits the model to data (vectorized, PyTorch, with progress logging and profiling).

        Args:
            data: An mxn tensor of training data
            learning_rate: Learning rate for each gradient step performed on a single entity-item sample.
            epochs: Number of training epochs, where each the model is trained on the cardinality of the dataset in each epoch.
            regularization_coefficient: Coefficient on the L2 regularization term.
            method: The prediction method to use.

        Mutates:
            The recommender to the new trained state.
        """
        if device is not None:
            self.device = device
        if isinstance(data, torch.Tensor):
            if data.is_sparse:
                data = data.to_dense()
            data = data.to(self.device)
        elif hasattr(data, 'toarray'):
            data = torch.tensor(data.toarray(), dtype=torch.float32, device=self.device)
        else:
            data = torch.tensor(np.array(data), dtype=torch.float32, device=self.device)

        # Reduce epochs for debugging
        if debug:
            debug_epochs = min(epochs, 5)
            if epochs > 5:
                print(f"[BPRRecommender] Debug mode: reducing epochs from {epochs} to {debug_epochs}")
            epochs = debug_epochs

        m, n = data.shape
        num_samples_per_epoch = m * n
        batch_size = 1024  # Vectorized batch size

        start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            # Vectorized negative sampling
            all_u, all_i, all_j = self._sample_dataset(data, num_samples=num_samples_per_epoch)

            num_batches = math.ceil(num_samples_per_epoch / batch_size)
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, num_samples_per_epoch)
                bu = all_u[batch_start:batch_end]
                bi = all_i[batch_start:batch_end]
                bj = all_j[batch_start:batch_end]

                # Gather embeddings
                U_bu = self._U[bu, :]  # (batch, k)
                V_bi = self._V[bi, :]  # (batch, k)
                V_bj = self._V[bj, :]  # (batch, k)

                x_ui = torch.sum(U_bu * V_bi, dim=1)  # (batch,)
                x_uj = torch.sum(U_bu * V_bj, dim=1)  # (batch,)
                x_uij = x_ui - x_uj  # (batch,)

                sigmoid_derivative = torch.exp(-x_uij) / (1 + torch.exp(-x_uij))  # (batch,)

                d_w = V_bi - V_bj  # (batch, k)
                d_hi = U_bu  # (batch, k)
                d_hj = -U_bu  # (batch, k)

                # Update U
                self._U[bu, :] += learning_rate * (
                    sigmoid_derivative.unsqueeze(1) * d_w - regularization_coefficient * U_bu
                )
                # Update V[i]
                self._V[bi, :] += learning_rate * (
                    sigmoid_derivative.unsqueeze(1) * d_hi - regularization_coefficient * V_bi
                )
                # Update V[j]
                self._V[bj, :] += learning_rate * (
                    sigmoid_derivative.unsqueeze(1) * d_hj - regularization_coefficient * V_bj
                )

            epoch_end = time.time()
            print(f"[BPRRecommender] Epoch {epoch+1}/{epochs} completed in {epoch_end-epoch_start:.2f}s")
        total_time = time.time() - start_time
        print(f"[BPRRecommender] Training completed in {total_time:.2f}s for {epochs} epochs.")


    def evaluate(
        self,
        test_data,
        method: PredictionMethod = PredictionMethod.DOT,
        device=None) -> float:
        """Evaluates the solution"""

        if device is not None:
            self.device = device
        if isinstance(test_data, torch.Tensor):
            if test_data.is_sparse:
                test_data = test_data.to_dense()
            test_data = test_data.to(self.device)
        elif hasattr(test_data, 'toarray'):
            test_data = torch.tensor(test_data.toarray(), dtype=torch.float32, device=self.device)
        else:
            test_data = torch.tensor(np.array(test_data), dtype=torch.float32, device=self.device)

        pred = self.predict_tensor(method)
        indices = torch.nonzero(test_data)
        predictions = pred[indices[:, 0], indices[:, 1]]
        true_values = test_data[indices[:, 0], indices[:, 1]]
        mse = torch.mean((true_values - predictions) ** 2).item()
        return mse


    def _calculate_sample_user_probability(self, data: torch.Tensor) -> torch.Tensor:
        """Gets the sample probability for each user.

        Args:
            data: An mxn tensor of observations on the device.

        Returns:
            A length m tensor containing the probability of sampling each entity.
        """
        m, n = data.shape
        assert m > 0
        data = torch.nan_to_num(data)

        observations_per_user = torch.sum(data, dim=1)
        samples_per_user = observations_per_user * (n - observations_per_user)
        total = samples_per_user.sum()
        if total == 0:
            return torch.full((m,), 1.0 / m, device=self.device)
        return samples_per_user / total


    def _predict_for_single_entry(self, u, i) -> float:
        """Predicts the value for a single user-item pair."""
        return torch.dot(self._U[u, :], self._V[i, :]).item()


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
        predictions = self.predict_tensor(method)
        return predictions.detach().cpu().numpy()

    def predict_tensor(
        self,
        method: PredictionMethod = PredictionMethod.DOT,
    ) -> torch.Tensor:
        """Gets the model predictions as a torch tensor."""
        self._checkrep()
        return calculate_predicted_matrix(self._U, self._V, method).to(self.device)


    def predict_new_entity(
        self,
        entity,
        learning_rate: float,
        epochs: int,
        regularization_coefficient: float,
        method: PredictionMethod = PredictionMethod.DOT,
        **kwargs) -> np.array:
        """
        Recommends items to an unseen entity. Robust to input shape and errors.
        Ensures entity is always 2D (m, n) for _sample_dataset compatibility.
        GPU compatible and accurate.
        """
        if torch.is_tensor(entity):
            tensor_entity = entity.to(dtype=torch.float32, device=self.device)
            if tensor_entity.is_sparse:
                tensor_entity = tensor_entity.to_dense()
        elif hasattr(entity, 'toarray'):
            tensor_entity = torch.tensor(
                entity.toarray(), dtype=torch.float32, device=self.device
            )
        else:
            tensor_entity = torch.tensor(
                np.array(entity), dtype=torch.float32, device=self.device
            )

        if tensor_entity.ndim == 1:
            tensor_entity = tensor_entity.reshape(1, -1)
        elif tensor_entity.ndim != 2:
            raise ValueError(f"Entity must be 1D or 2D, got shape {tuple(tensor_entity.shape)}")

        num_iterations = int(
            epochs * tensor_entity.shape[0] * tensor_entity.shape[1]
        )
        embedding_dim = self._U.shape[1]
        init_stddev = math.sqrt(1 / embedding_dim)
        new_entity_embedding = torch.normal(
            mean=0.0,
            std=init_stddev,
            size=(embedding_dim,),
            device=self.device,
        )

        if num_iterations > 0:
            try:
                _, all_i, all_j = self._sample_dataset(
                    tensor_entity, num_samples=num_iterations
                )
            except Exception as e:
                print(
                    f"[BPRRecommender][ERROR] _sample_dataset failed: {e}. "
                    f"entity shape: {tuple(tensor_entity.shape)}"
                )
                return np.zeros(self._V.shape[0])

            for iteration_count in range(num_iterations):
                i = all_i[iteration_count]
                j = all_j[iteration_count]

                x_ui = torch.dot(new_entity_embedding, self._V[i])
                x_uj = torch.dot(new_entity_embedding, self._V[j])
                x_uij = x_ui - x_uj

                sigmoid_derivative = torch.exp(-x_uij) / (1 + torch.exp(-x_uij))

                d_w = self._V[i] - self._V[j]
                new_entity_embedding += learning_rate * (
                    sigmoid_derivative * d_w
                    - regularization_coefficient * new_entity_embedding
                )

        predictions_tensor = calculate_predicted_matrix(
            new_entity_embedding.unsqueeze(0), self._V, method
        )
        return np.squeeze(predictions_tensor.detach().cpu().numpy())


Recommender.register(BPRRecommender)
