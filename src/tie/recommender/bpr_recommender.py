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
        assert num_samples > 0
        # Ensure data is a torch tensor on the correct device
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(data):
            data = torch.tensor(np.array(data), dtype=torch.float32, device=self.device)
        else:
            data = data.to(self.device)

        m, n = data.shape
        sample_user_probability = torch.tensor(self._calculate_sample_user_probability(data.cpu().numpy()), dtype=torch.float32, device=self.device)
        num_items_per_user = torch.sum(data, dim=1).float()
        num_items_per_user[num_items_per_user == 0.0] = float('nan')
        assert num_items_per_user.shape[0] == m
        sample_item_probability = torch.nan_to_num(data / num_items_per_user.unsqueeze(1))
        joint_user_item_probability = sample_user_probability.unsqueeze(1) * sample_item_probability
        assert joint_user_item_probability.shape == (m, n)
        flattened_probability = joint_user_item_probability.flatten()
        # Move to CPU for numpy random choice
        flattened_probability_np = flattened_probability.cpu().numpy()
        u_i = np.random.choice(np.arange(m * n), size=(num_samples,), p=flattened_probability_np)
        all_u = u_i // n
        all_i = u_i % n
        non_observations = (1 - data).cpu().numpy()
        unique_users, counts = np.unique(all_u, return_counts=True)
        value_to_count = dict(zip(unique_users, counts))
        u_to_j = {}
        for u, count in value_to_count.items():
            potential_j = non_observations[u, :]
            potential_j_sum = np.sum(potential_j)
            if potential_j_sum == 0:
                # If no zero entries, fallback to uniform
                all_j_for_user = np.random.choice(n, size=count, replace=True)
            else:
                all_j_for_user = np.random.choice(n, size=count, replace=True, p=potential_j / potential_j_sum)
            u_to_j[u] = list(all_j_for_user)
        all_j = []
        for u in all_u:
            j = u_to_j[u].pop()
            all_j.append(j)
        assert len(all_u) == len(all_j) == len(all_i)
        return np.array(all_u), np.array(all_i), np.array(all_j)



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
            data_np = data.cpu().numpy()
            all_u, all_i, all_j = self._sample_dataset(data_np, num_samples=num_samples_per_epoch)

            # Convert to torch tensors
            all_u = torch.tensor(all_u, dtype=torch.long, device=self.device)
            all_i = torch.tensor(all_i, dtype=torch.long, device=self.device)
            all_j = torch.tensor(all_j, dtype=torch.long, device=self.device)

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

        pred_np = self.predict(method)
        if isinstance(pred_np, torch.Tensor):
            pred = pred_np.detach().clone().to(torch.float32).to(self.device)
        else:
            pred = torch.tensor(pred_np, dtype=torch.float32, device=self.device)
        indices = torch.nonzero(test_data)
        predictions = pred[indices[:, 0], indices[:, 1]]
        true_values = test_data[indices[:, 0], indices[:, 1]]
        mse = torch.mean((true_values - predictions) ** 2).item()
        return mse
        flattened_probability = joint_user_item_probability.flatten("C")
        u_i = np.random.choice(
            np.arange(m * n), size=(num_samples,), p=flattened_probability
        )

        all_u = u_i // n
        all_i = u_i % n
        assert (all_i < 611).all()

        non_observations = 1 - data

        unique_users, counts = np.unique(all_u, return_counts=True)
        value_to_count = dict(zip(unique_users, counts))

        u_to_j = {}

        # for each u
        for u, count in value_to_count.items():
            # get
            potential_j = non_observations[u, :]

            all_j_for_user = np.random.choice(
                n, size=count, replace=True, p=potential_j / np.sum(potential_j)
            )

            u_to_j[u] = all_j_for_user.tolist()

        all_j = []

        for u in all_u:
            j = u_to_j[u].pop()
            all_j.append(j)

        assert len(all_u) == len(all_j) == len(all_i)

        return all_u, all_i, all_j


    def _calculate_sample_user_probability(self, data: np.ndarray) -> np.array:
        """Gets the sample probability for each user.

        Args:
            data: An mxn matrix of observations.

        Returns:
            A length m array containing the probability of sampling each entity.
        """
        m, n = data.shape
        data = np.nan_to_num(data)

        observations_per_user = np.sum(data, axis=1)
        assert observations_per_user.shape == (m,)

        samples_per_user = observations_per_user * (n - observations_per_user)
        sample_user_probability = samples_per_user / np.sum(samples_per_user)
        assert sample_user_probability.shape == (m,)

        return sample_user_probability


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
        # Ensure tensors are on CPU before passing to numpy-based utils
        U_cpu = self._U.cpu() if self._U.is_cuda else self._U
        V_cpu = self._V.cpu() if self._V.is_cuda else self._V
        return calculate_predicted_matrix(U_cpu, V_cpu, method)


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
        # Accepts either a PyTorch sparse tensor, dense tensor, or numpy array
        if isinstance(entity, torch.Tensor):
            if entity.is_sparse:
                entity = entity.to_dense()
            entity = entity.cpu().numpy()
        elif hasattr(entity, 'toarray'):
            entity = entity.toarray()
        else:
            entity = np.array(entity)

        # Ensure entity is 2D (m, n) for _sample_dataset
        if entity.ndim == 1:
            entity_reshaped = entity.reshape(1, -1)
        elif entity.ndim == 2:
            entity_reshaped = entity
        else:
            raise ValueError(f"Entity must be 1D or 2D, got shape {entity.shape}")

        # Ensure self._V is on CPU for numpy ops
        V_cpu = self._V.cpu().numpy() if self._V.is_cuda else self._V.numpy()

        num_iterations = epochs * entity_reshaped.shape[0] * entity_reshaped.shape[1]

        new_entity_embedding = np.random.normal(
            loc=0, scale=math.sqrt(1 / self._U.shape[1]), size=(1, self._U.shape[1])
        )

        try:
            _, all_i, all_j = self._sample_dataset(entity_reshaped, num_samples=num_iterations)
        except Exception as e:
            print(f"[BPRRecommender][ERROR] _sample_dataset failed: {e}. entity_reshaped shape: {entity_reshaped.shape}")
            # Return zeros for predictions if sampling fails
            return np.zeros(V_cpu.shape[0])

        for iteration_count in range(num_iterations):
            i = all_i[iteration_count]
            j = all_j[iteration_count]

            x_ui = np.dot(new_entity_embedding, V_cpu[i, :])
            x_uj = np.dot(new_entity_embedding, V_cpu[j, :])
            x_uij = x_ui - x_uj

            sigmoid_derivative = (math.e ** (-x_uij)) / (1 + math.e ** (-x_uij))

            d_w = V_cpu[i, :] - V_cpu[j, :]

            new_entity_embedding += learning_rate * (
                sigmoid_derivative * d_w
                - (regularization_coefficient * new_entity_embedding)
            )

        return np.squeeze(
            calculate_predicted_matrix(new_entity_embedding, V_cpu, method)
        )


Recommender.register(BPRRecommender)
