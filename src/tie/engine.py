import copy

import numpy as np
import pandas as pd
import torch
import concurrent.futures
import copy
from .constants import PredictionMethod
from .exceptions import TechniqueNotFoundException
from .matrix import ReportTechniqueMatrix
from .recommender import Recommender
from .utils import (
    get_mitre_technique_ids_to_names,
    normalized_discounted_cumulative_gain,
    normalized_discounted_cumulative_gain_tensor,
    precision_at_k,
    precision_at_k_tensor,
    recall_at_k,
    recall_at_k_tensor,
)

def parameter_cartesian_product(
    variables_names: tuple[str],
    values: tuple[tuple[float]]):
    """Yield cartesian product of all variables.

    Args:
        variable_names: iterable of variables for which to generate
            combinations.
        values: the values over which to generate combinations such that
            values[i] contains all values for variables[i].

    Yields:
        A dictionary mapping each variable to a value such that each dictionary
        yielded is a unique combination of the cartesian product of values over
        variables.
    """
    assert len(variables_names) == len(values)

    # base case: No variables over which to make product

    if len(variables_names) == 0:
        yield {}
    else:
        for value in values[0]:
            for remaining_parameters in parameter_cartesian_product(
                variables_names[1:], values[1:]
            ):
                yield remaining_parameters | {variables_names[0]: value}

class TechniqueInferenceEngine:
    """A technique inference engine.

    The technique inference engine predicts, given a bag of MITRE
    ATT&CK techniques, the next most likely techniques that would be part
    of that report, given the report dataset provided.
    
    Abstraction function:
    	AF(training_data, test_data, model, enterprise_attack_filepath) =
          a technique inference engine to be trained using model on
          training_data and evaluated on test_data
          according to the MITRE ATT&CK framework specified in
          enterprise_attack_filepath.
    
    Rep invariant:
     - training_data.shape == test_data.shape == validation_data.shape
     - model is not None
     - prediction_method is not None
     - len(enterprise_attack_filepath) >= 0
    
    Safety from rep exposure:
     - all attributes are private
     - training_data and test_data are immutable
     - model is deep copied and never returned
    """
    def __init__(
        self,
        training_data: ReportTechniqueMatrix,
        validation_data: ReportTechniqueMatrix,
        test_data: ReportTechniqueMatrix,
        model: Recommender,
        prediction_method: PredictionMethod,
        enterprise_attack_filepath: str,
    ):
        """Initializes a TechniqueInferenceEngine object.

        Args:
            training_data: the data on which to train the model.
            test_data: the data on which to evaluate the model's performance.
            model: the model to train.
            prediction_method: the method to use for predictions.
            enterprise_attack_filepath: filepath for the MITRE enterprise ATT&CK json
                information.
        """
        self._enterprise_attack_filepath = enterprise_attack_filepath

        self._training_data = training_data
        self._validation_data = validation_data
        self._test_data = test_data
        self._model = copy.deepcopy(model)
        self._prediction_method = prediction_method

        self._checkrep()

    def _checkrep(self):
        """Asserts the rep invariant."""
        # - training_data.shape == test_data.shape == validation_data.shape
        assert (
            self._training_data.shape
            == self._test_data.shape
            == self._validation_data.shape
        )
        # - model is not None
        assert self._model is not None
        # - prediction_method is not None
        assert self._prediction_method is not None
        # - len(enterprise_attack_filepath) >= 0
        assert len(self._enterprise_attack_filepath) >= 0

    def _add_technique_name_to_dataframe(self, data: pd.DataFrame):
        """Adds a technique name column to the dataframe.

        Args:
            data: data indexed by technique id.

        Mutates:
            data to add a column titled "technique_name"
        """
        all_mitre_technique_ids_to_names = get_mitre_technique_ids_to_names(
            self._enterprise_attack_filepath
        )
        data.loc[:, "technique_name"] = data.apply(
            lambda row: all_mitre_technique_ids_to_names.get(row.name), axis=1
        )

    def fit(self, **kwargs) -> float:
        """Fit the model to the data.

        Kwargs: Model specific args.

        Returns:
            The MSE of the prediction matrix, as determined by the test set.
        """
        # train
        self._model.fit(self._training_data.to_sparse_tensor(), **kwargs)

        mean_squared_error = self._model.evaluate(
            self._test_data.to_sparse_tensor(), method=self._prediction_method
        )

        self._checkrep()
        return mean_squared_error

    def fit_with_validation(self, **kwargs) -> dict[str, float]:
        """Fits the model by validating hyperparameters on the cross validation data.

        Selects the hyperparameters which maximize normalized discounted cumulative gain
        (NDCG) on the validation data.

        Args:
            kwargs: mapping of hyperparameter to values over which to cross-validate.

        Returns:
            A mapping of each kwarg to the value from the best hyperparameter
            combination.
        """
        def fit_and_score(hyperparameters, device):
            # Deep copy engine and model to avoid cross-process issues
            engine_copy = copy.deepcopy(self)
            # Move model and data to the assigned device
            if hasattr(engine_copy._model, 'to'):
                engine_copy._model = engine_copy._model.to(device)
            if hasattr(engine_copy._training_data, 'to_tensor'):
                engine_copy._training_data = engine_copy._training_data.to_tensor(device)
            if hasattr(engine_copy._validation_data, 'to_tensor'):
                engine_copy._validation_data = engine_copy._validation_data.to_tensor(device)
            if hasattr(engine_copy._test_data, 'to_tensor'):
                engine_copy._test_data = engine_copy._test_data.to_tensor(device)
            print(f"[fit_with_validation] Running on device {device} with hyperparameters: {hyperparameters}")
            engine_copy.fit(**hyperparameters)
            predictions = engine_copy.predict_tensor(device)
            validation_tensor = engine_copy._validation_data.to_dense_tensor(
                device=predictions.device
            )
            score = recall_at_k_tensor(predictions, validation_tensor, k=20)
            print(f"[fit_with_validation] Score for {hyperparameters} on device {device}: {score}")
            return (hyperparameters, score)

        # Detect available GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_list = [torch.device(f'cuda:{i}') for i in range(num_gpus)] if num_gpus > 0 else [torch.device('cpu')]
        
        best_hyperparameters = {}
        best_score = -float("inf")

        variable_names = tuple(kwargs.keys())
        variable_values = tuple(kwargs.get(key) for key in variable_names)

        total_combinations = 1
        for vals in variable_values:
            total_combinations *= len(vals)

        # Prepare all hyperparameter combinations
        combos = list(parameter_cartesian_product(variable_names, variable_values))

        # Use ThreadPoolExecutor for parallelism
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_list)) as executor:
            future_to_combo = {}
            for idx, hyperparameters in enumerate(combos):
                device = device_list[idx % len(device_list)]
                future = executor.submit(fit_and_score, hyperparameters, device)
                future_to_combo[future] = hyperparameters
            for future in concurrent.futures.as_completed(future_to_combo):
                hyperparameters, score = future.result()
                results.append((hyperparameters, score))
                if score > best_score:
                    best_score = score
                    best_hyperparameters = hyperparameters
        print(f"[fit_with_validation] Best hyperparameters: {best_hyperparameters} with score {best_score}")
        self.fit(**best_hyperparameters)
        return best_hyperparameters

    def precision(self, k: int = 10) -> float:
        r"""Calculates the precision of the top k model predictions.

        Precision is defined as the average fraction of items in the top k predictions
        which appear in the test set.  If k < the number of items in the test set for a
        particular user, then the maximum precision is 1.0.

        Mathematically, it is defined as
        precision@k = (1\m) \sum_u (\sum_{i=1}^k [[pred_i in test set]] / k)

        Args:
            k: the number of predictions to include in the top k.  Requires 0 < k <= n.

        Returns:
            The computed precision for the top k model predictions.
        """
        predictions = self.predict_tensor()
        test_tensor = self._test_data.to_dense_tensor(device=predictions.device)
        return precision_at_k_tensor(predictions, test_tensor, k)

    def recall(self, k: int = 10) -> float:
        r"""Calculates the recall of the top k model predictions.

        Recall is defined as the average fraction of items in the test set which appear
        in the top k predictions.  If k >= the number of items in the test set for a
        particular user, then the maximum recall is 1.0.

        Mathematically, it is defined as
        recall@k =
            (1\m) \sum_u (\sum_{i=1}^k [[pred_i in test set]] / |test set for entity i|

        Args:
            k: the number of predictions to include in the top k.  Requires k > 0.

        Returns:
            The computed recall for the top k model predictions.
        """
        predictions = self.predict_tensor()
        test_tensor = self._test_data.to_dense_tensor(device=predictions.device)
        return recall_at_k_tensor(predictions, test_tensor, k)

    def normalized_discounted_cumulative_gain(self, k: int = 10) -> float:
        r"""Computes the Normalized Discounted Cumulative Gain (NDCG) on the test set.

        NDCG measures the goodness of a ranking based on the relative ordering of
        test set entries in the top-k predictions of the model.  Test set predictions
        that appear near the top of the top-k predictions (in descending order)
        for the model contribute more to NDCG than predictions which are ranked lower.
        NDCG ranges from 0 to 1, where 1 is a perfect ranking.

        Mathematically, NDCG is defined as
        NDCG@K = DCG@K / IDCG@K
        DCG@K = (1/m) \sum_u \sum_{i=1}^k (2^[[pred_i in test set]] - 1)/(log_2 (i+1))
        IDCG@K is a normalization constant corresponding to the maximum possible value
            of DCG@K

        Args:
            k: the number of predictions to get from the model.

        Returns:
            NDCG computed on the top k predictions.
        """
        predictions = self.predict_tensor()
        test_tensor = self._test_data.to_dense_tensor(device=predictions.device)
        return normalized_discounted_cumulative_gain_tensor(predictions, test_tensor, k=k)

    def predict_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        """Gets the predictions as a torch tensor, keeping them on device when possible."""

        if hasattr(self._model, "predict_tensor"):
            predictions = self._model.predict_tensor(method=self._prediction_method)
        else:
            predictions = torch.as_tensor(
                self._model.predict(method=self._prediction_method), dtype=torch.float32
            )
        if device is not None:
            predictions = predictions.to(device)
        return predictions

    def predict_tensor_for_new_report(
        self,
        techniques: frozenset[str],
        device: torch.device | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Builds predictions for techniques not seen during training as a tensor."""

        all_technique_ids = self._training_data.technique_ids
        technique_ids_to_indices = {
            all_technique_ids[i]: i for i in range(len(all_technique_ids))
        }
        technique_indices: list[int] = []
        seen_indices: set[int] = set()
        for technique in techniques:
            if technique not in technique_ids_to_indices:
                raise TechniqueNotFoundException(
                    f"Model has not been trained on {technique}."
                )
            idx = technique_ids_to_indices[technique]
            if idx not in seen_indices:
                technique_indices.append(idx)
                seen_indices.add(idx)
        if not technique_indices:
            raise TechniqueNotFoundException(
                "At least one technique must be provided for inference."
            )
        technique_indices.sort()

        n = self._training_data.n
        indices_2d = np.array([[0, idx] for idx in technique_indices], dtype=np.int64)
        values = np.ones((len(technique_indices),), dtype=np.float32)
        indices_tensor = torch.tensor(indices_2d, dtype=torch.long)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        technique_tensor = torch.sparse_coo_tensor(
            indices_tensor.t(), values_tensor, size=(1, n)
        )

        target_device = device
        if target_device is None:
            model_device = getattr(self._model, "device", None)
            target_device = (
                model_device if isinstance(model_device, torch.device) else torch.device("cpu")
            )

        predictions_vec = self._model.predict_new_entity(
            technique_tensor, method=self._prediction_method, **kwargs
        )
        if predictions_vec is None:
            predictions_tensor = torch.empty(0, dtype=torch.float32, device=target_device)
        else:
            predictions_tensor = torch.as_tensor(
                np.asarray(predictions_vec), dtype=torch.float32, device=target_device
            ).reshape(-1)

        full_predictions = torch.zeros(n, dtype=torch.float32, device=target_device)
        if predictions_tensor.numel() == 0:
            return full_predictions
        if predictions_tensor.numel() == n:
            full_predictions.copy_(predictions_tensor[:n])
            return full_predictions
        if predictions_tensor.numel() == len(technique_indices):
            for i, idx in enumerate(technique_indices):
                if 0 <= idx < n:
                    try:
                        full_predictions[idx] = predictions_tensor[i]
                    except Exception as exc:
                        print(
                            f"[predict_tensor_for_new_report] Could not assign predictions_tensor[{i}] to full_predictions[{idx}]: {exc}"
                        )
            return full_predictions
        min_len = min(predictions_tensor.numel(), len(technique_indices))
        for i in range(min_len):
            idx = technique_indices[i]
            if 0 <= idx < n:
                try:
                    full_predictions[idx] = predictions_tensor[i]
                except Exception as exc:
                    print(
                        f"[predict_tensor_for_new_report] Could not assign predictions_tensor[{i}] to full_predictions[{idx}]: {exc}"
                    )
        if predictions_tensor.numel() != len(technique_indices):
            print(
                f"[predict_tensor_for_new_report] Ambiguous mapping between predictions_tensor (len={predictions_tensor.numel()}) "
                f"and technique_indices (len={len(technique_indices)}). Only mapped up to min length."
            )
        return full_predictions

    def predict(self) -> pd.DataFrame:
        """Obtains model predictions.

        For each report, predicts a value for every technique based on the likelihood
        that technique should be featured in the report.  A higher predicted value for
        technique a than technique b represents an inference that technique a is more
        likely in the report than technique b.

        Returns:
            A dataframe with the same shape, index, and columns as training_data and
            test_data containing the predictions values for each report and technique
            combination.
        """
        predictions_tensor = self.predict_tensor()
        predictions = predictions_tensor.detach().cpu().numpy()

        predictions_dataframe = pd.DataFrame(
            predictions,
            index=self._training_data.report_ids,
            columns=self._training_data.technique_ids,
        )

        self._checkrep()
        return predictions_dataframe

    def view_prediction_performance_table_for_report(
        self,
        report_id: int) -> pd.DataFrame:
        """Gets the training data, test data, and predictions for a particular report.

        Args:
            report_id: identifier for the report.  Must be in the training_data and
                test_data.

        Returns:
            A length len(training_data) dataframe indexed by technique id containing the
            following columns:
                - predictions, the predicted value for that Technique
                - training_data: 1 if technique was present in the input, 0 otherwise
                - test_data: all 0's since no test data for cold start predictions
                - technique_name: the technique name for the identifying technique in
                  the index
        """
        report_data = pd.DataFrame(
            {
                "predictions": self.predict().loc[report_id],
                "training_data": self._training_data.to_pandas().loc[report_id],
                "test_data": self._test_data.to_pandas().loc[report_id],
            }
        )

        # add name for convenience
        self._add_technique_name_to_dataframe(report_data)

        self._checkrep()
        return report_data

    def predict_for_new_report(
        self, techniques: frozenset[str], **kwargs) -> pd.DataFrame:
        """Predicts for a new, yet-unseen report.

        Args:
            techniques: an iterable of MITRE technique identifiers involved
                in the new report.

        Returns:
            A length n dataframe indexed by technique id containing the following
            columns:
                - predictions, the predicted value for that Technique
                - training_data: 1 if technique was present in the input, 0 otherwise
                - test_data: all 0's since no test data for cold start predictions
                - technique_name: the technique name for the identifying technique in
                  the index
        """
        # need to turn into the embeddings in the original matrix
        all_technique_ids = self._training_data.technique_ids
        technique_ids_to_indices = {
            all_technique_ids[i]: i for i in range(len(all_technique_ids))
        }

        technique_indices = set()
        for technique in techniques:
            if technique in technique_ids_to_indices:
                technique_indices.add(technique_ids_to_indices[technique])
            else:
                raise TechniqueNotFoundException(
                    f"Model has not been trained on {technique}."
                )

        technique_indices = list(technique_indices)
        technique_indices.sort()
        n = self._training_data.n
        predictions_tensor = self.predict_tensor_for_new_report(
            techniques, **kwargs
        )
        full_predictions = predictions_tensor.detach().cpu().numpy()
        training_indices_dense = np.zeros(n)
        for idx in technique_indices:
            if 0 <= idx < n:
                training_indices_dense[idx] = 1
        result_dataframe = pd.DataFrame(
            {
                "predictions": full_predictions,
                "training_data": training_indices_dense,
                "test_data": np.zeros(n),
            },
            index=all_technique_ids,
        )

        self._add_technique_name_to_dataframe(result_dataframe)

        self._checkrep()
        return result_dataframe

    def get_U(self) -> np.ndarray:
        """Get the item embeddings of the model."""
        return self._model.U

    def get_V(self) -> np.ndarray:
        """Get the user embeddings of the model."""
        return self._model.V
