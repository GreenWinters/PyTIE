'''
Top Items Recommender 


Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from .recommender import Recommender


class TopItemsRecommender(Recommender):
    """
    A recommender model which always recommends the most observed techniques.

    A recommender model which always recommends the most observed techniques in the
    dataset in frequency order.

    Abstraction function:
       AF(m, n, item_frequencies) = a recommender model which recommends the n
           items in order of frequency according to item_frequencies
           for each of the m entities
    
    Rep invariant:
       - m > 0
       - n > 0
       - item_frequencies.shape == (n,)
       - 0 <= item_frequencies[i] <= n-1 for all 0 <= i < n
    
    Safety from rep exposure:
       - m and n are private and immutable
       - item_frequency is private and never returned
    """
    def __init__(self, m, n, k, device=None):
        """Initializes a TopItemsRecommender object."""
        self._m = m  # entity dimension
        self._n = n  # item dimension
        self.device = device if device is not None else torch.device('cpu')
        # array of item frequencies,
        # ranging from 0 (least frequent) to n-1 (most frequent)
        self._item_frequencies = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self._checkrep()

    def to(self, device):
        self.device = device
        self._item_frequencies = self._item_frequencies.to(device)
        return self

    def _checkrep(self):
        """Asserts the rep invariant."""
        #   - m > 0
        assert self._m > 0
        #   - n > 0
        assert self._n > 0
        #   - item_frequencies.shape == (n,)
        assert self._item_frequencies.shape == (self._n,)
        #   - 0 <= item_frequencies[i] <= n-1 for all 0 <= i < n
        assert (0 <= self._item_frequencies).all()
        assert (self._item_frequencies <= self._n - 1).all()

    def U(self) -> np.ndarray:
        """Gets U as a factor of the factorization UV^T."""
        raise NotImplementedError

    def V(self) -> np.ndarray:
        """Gets V as a factor of the factorization UV^T."""
        raise NotImplementedError

    def _scale_item_frequency(self, item_frequencies: np.array) -> np.array:
        """Scales the item frequencies from 0 to 1.

        Assigns each item the value 1/(n-1) * rank_i, where rank_i is the rank
        of item i in sorted ascending order by frequency.
        Therefore, the top frequency item will take scaled value 1, while the least
        frequent item will take scaled value 0.

        Args:
            item_frequencies: A length-n vector containing the number of occurrences
                of each item in the dataset.

        Returns:
            A scaled version of item_frequencies.
        """
        # assert 1d array
        assert len(item_frequencies.shape) == 1

        scaled_ranks = item_frequencies / (len(item_frequencies) - 1)

        assert scaled_ranks.shape == item_frequencies.shape

        self._checkrep()
        return scaled_ranks

    def fit(self, data, device=None, **kwargs):
        # Accepts either numpy array, PyTorch tensor, or dict with indices/values/shape
        if hasattr(data, 'indices') and hasattr(data, 'values') and hasattr(data, 'shape'):
            technique_matrix = np.zeros((self._m, self._n))
            # If data.indices is a method, call it; if it's an attribute, use it directly
            # Ensure PyTorch sparse tensor is coalesced before accessing indices
            tensor_data = data
            if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                tensor_data = tensor_data.coalesce()
            indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
            horizontal_indices = tuple(indices[:, 0])
            vertical_indices = tuple(indices[:, 1])
            vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
            if torch.is_tensor(vals):
                vals = vals.cpu().numpy()
            technique_matrix[horizontal_indices, vertical_indices] = vals
        elif isinstance(data, np.ndarray):
            technique_matrix = data
        elif torch.is_tensor(data):
            technique_matrix = data.cpu().numpy()
        else:
            raise ValueError("Unsupported data format for fit().")

        technique_frequency = torch.tensor(technique_matrix.sum(axis=0), dtype=torch.float32, device=self.device)
        assert technique_frequency.shape[0] == self._n
        ranks = torch.argsort(torch.argsort(technique_frequency))
        self._item_frequencies = ranks.to(self.device)
        self._checkrep()

    def evaluate(self, test_data, **kwargs) -> float:
        predictions_matrix = self.predict()
        if hasattr(test_data, 'indices') and hasattr(test_data, 'values'):
            tensor_data = test_data
            if hasattr(tensor_data, 'is_sparse') and tensor_data.is_sparse:
                tensor_data = tensor_data.coalesce()
            indices = tensor_data.indices() if callable(tensor_data.indices) else tensor_data.indices
            if torch.is_tensor(indices):
                indices = indices.t().cpu().numpy()
            row_indices = tuple(indices[:, 0])
            column_indices = tuple(indices[:, 1])
            prediction_values = predictions_matrix[row_indices, column_indices]
            vals = tensor_data.values() if callable(tensor_data.values) else tensor_data.values
            if torch.is_tensor(vals):
                vals = vals.cpu().numpy()
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

    def predict(self, device=None, **kwargs) -> np.ndarray:
        scaled_ranks = self._scale_item_frequency(self._item_frequencies)
        matrix = scaled_ranks.repeat(self._m).reshape(self._n, self._m).T
        assert matrix.shape == (self._m, self._n)
        self._checkrep()
        return matrix.cpu().numpy() if matrix.device.type != 'cpu' else matrix.numpy()

    def predict_new_entity(self, entity, device=None, **kwargs) -> np.array:
        self._checkrep()
        scaled = self._scale_item_frequency(self._item_frequencies)
        return scaled.cpu().numpy() if scaled.device.type != 'cpu' else scaled.numpy()
