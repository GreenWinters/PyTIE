'''
Technique Inference Engine (TIE)

This script allows you to train, run, and experiment with the Technique Inference Engine.
For more information, visit:
https://center-for-threat-informed-defense.github.io/technique-inference-engine/

Modified by: @GreenWinters
Based on original code from: https://github.com/center-for-threat-informed-defense/technique-inference-engine
Significant changes made for research/development purposes.
See LICENSE and README for details.
'''
import argparse
import json
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.manifold
import src.tie.recommender
import torch
from src.tie.constants import PredictionMethod
from src.tie.engine import TechniqueInferenceEngine
from src.tie.matrix_builder import ReportTechniqueMatrixBuilder
from src.tie.recommender import (
    BPRRecommender,
    FactorizationRecommender,
    ImplicitBPRRecommender,
    ImplicitWalsRecommender,
    Recommender,
    TopItemsRecommender,
    WalsRecommender,
)


def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def save_metrics_json(metrics, model_name, timestamp):
    os.makedirs("tie_model", exist_ok=True)
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join("tie_model", filename)
    metrics_converted = convert_types(metrics)
    with open(filepath, "w") as f:
        json.dump(metrics_converted, f, indent=2)
    print(f"Metrics saved to {filepath}")


def save_model(model, method, model_class_name, directory="tie_model"):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_class_name}_{method}_{timestamp}.pt"
    filepath = os.path.join(directory, filename)
    # If model is a PyTorch nn.Module or has state_dict, save state_dict
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), filepath)
    else:
        torch.save(model, filepath)
    print(f"Model saved to {filepath}")


def is_pytorch_model(model):
    """
    Returns True if the model is a PyTorch nn.Module or supports .to(device).
    """
    return hasattr(model, 'to') and callable(getattr(model, 'to', None))


def get_device():
    """
    Returns the torch device to use for training (cuda if available, else cpu).
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        return torch.device("cuda")
    else:
        print("No GPU found, running on CPU.")
        return torch.device("cpu")


def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).to(device)
    elif hasattr(data, 'X'):  # Example for custom class
        return torch.tensor(data.X, dtype=torch.float32).to(device)
    else:
        return data  # If already tensor or compatible

def load_data(device):
    """
    Load and return training, test, validation data, and enterprise_attack_filepath, moved to the correct device if possible.
    """
    validation_ratio = 0.1
    test_ratio = 0.2

    # data locations
    dataset_filepath = "technique_inference_engine/data/combined_dataset_full_frequency.json"
    enterprise_attack_filepath = "technique_inference_engine/data/stix/enterprise-attack.json"
    # make data
    data_builder = ReportTechniqueMatrixBuilder(
        combined_dataset_filepath=dataset_filepath,
        enterprise_attack_filepath=enterprise_attack_filepath,
    )
    training_data, test_data, validation_data = data_builder.build_train_test_validation(test_ratio, validation_ratio)
    # Move to device if possible
    training_data = to_tensor(training_data)
    test_data = to_tensor(test_data)
    validation_data = to_tensor(validation_data)
    return training_data, test_data, validation_data, enterprise_attack_filepath


def test_multiple_embedding_dimensions(model_class: Recommender, method: PredictionMethod, training_data, validation_data, test_data, enterprise_attack_filepath, out_file: str, device=None, **kwargs):
    """Runs model_class at multiple embedding dimensions and saves results (PyTorch refactor)."""
    assert len(out_file) > 0
    results = []
    embedding_dimensions = (4,8,10,16,32,64)
    print(f"Starting embedding dimension sweep: {embedding_dimensions}")
    for embedding_dimension in embedding_dimensions:
        print(f"\n{model_class.__name__}: Training with embedding dimension: {embedding_dimension}")
        td = to_tensor(training_data)
        vd = to_tensor(validation_data)
        tsd = to_tensor(test_data)
        model = model_class(m=getattr(td, 'm', td.shape[0]), n=getattr(td, 'n', td.shape[1]), k=embedding_dimension, device=device)
        # Robust device transfer for model and data
        if device is not None:
            td = td.to(device) if hasattr(td, 'to') else td
            vd = vd.to(device) if hasattr(vd, 'to') else vd
            tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
            # Try to send model to device regardless of is_pytorch_model result
            try:
                if hasattr(model, 'to') and callable(getattr(model, 'to', None)):
                    model = model.to(device)
                    print(f"Model and data moved to device: {device}")
                else:
                    print(f"Warning: {model_class.__name__} does not support .to(device). Running on CPU.")
            except Exception as e:
                print(f"Error moving model to device: {e}. Running on CPU.")
        tie = TechniqueInferenceEngine(
            training_data=td,
            validation_data=vd,
            test_data=tsd,
            model=model,
            prediction_method=method,
            enterprise_attack_filepath=enterprise_attack_filepath)
        print(f"{model_class.__name__}: Starting validation sweep for embedding dimension {embedding_dimension}...")
        try:
            best_hyperparameters = tie.fit_with_validation(**kwargs)
            print(f"{model_class.__name__}: Finished validation sweep for embedding dimension {embedding_dimension}. Best hyperparameters: {best_hyperparameters}")
            run_stats = {
                "embedding_dimension": embedding_dimension,
                **best_hyperparameters
            }
            k_values = (10, 20, 50, 100)
            for k in k_values:
                run_stats[f"precision_at_{k}"] = tie.precision(k=k)
                run_stats[f"recall_at_{k}"] = tie.recall(k=k)
                run_stats[f"ndcg_at_{k}"] = tie.normalized_discounted_cumulative_gain(k=k)
            print(f"{model_class.__name__}: Metrics for embedding dimension {embedding_dimension}: {run_stats}")
            results.append(run_stats)
        except ValueError as ve:
            print(f"{model_class.__name__}: Skipping embedding dimension {embedding_dimension} due to error: {ve}")
            results.append({
                "embedding_dimension": embedding_dimension,
                "error": str(ve),
                "status": "skipped_singular_matrix"
            })
    print(f"{model_class.__name__}: All embedding dimension sweeps complete. Saving results to {out_file}")
    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_csv(out_file)


def make_tsne_embeddings(embeddings: np.ndarray):
    """
    Create 2D representation of embeddings using t-SNE.

    Args:
        embeddings: an mxk array of m embeddings in k-dimensional space.

    Returns:
        A tuple of the form (x_1, x_2) where x_1 and x_2 are length m
        such that (x_1[i], x_2[i]) is the 2-dimensional point cotnaining the 2-dimensional
        repsresentation for embeddings[i, :].
    """
    tsne = sklearn.manifold.TSNE(
        n_components=2,
        perplexity=15,
        learning_rate="auto",
        init='pca',
        verbose=True,
        n_iter=10000,
    )
    V_proj = tsne.fit_transform(embeddings)
    x = V_proj[:, 0]
    y = V_proj[:, 1]
    return x, y


def visualize_embeddings(tie):
    '''
    Visualize Embeddings

    Because the model's reports exist in an embedding dimension greater than 4, it 
    can be difficult to visualize how the reports are clustered. This cell uses the currently 
    trained model (from section 4) to plot a 2-dimensional representation of the model's 
    embeddings using t-SNE. In the chart below, reports that are far apart in the 4-dimensional 
    space are similarly distant in the 2-dimensional space.

    If you wish to visualize how techniques are clustered (instead of reports), switch:

    U = tie.get_U()
    x_1, x_2 = make_tsne_embeddings(U)
    to:

    V = tie.get_V()
    x_1, x_2 = make_tsne_embeddings(V)
    '''
    U = tie.get_U()
    x_1, x_2 = make_tsne_embeddings(U)
    plt.scatter(x_1, x_2, s=0.5)
    plt.xlabel("t-SNE x")
    plt.ylabel("t-SNE y")
    plt.title("t-SNE Visualization of Report Embeddings")
    plt.show()


def run_experiment(model_class, method, training_data, validation_data, test_data, enterprise_attack_filepath, embedding_dimension, k, hyperparameters, label=None, device=None):
    """
    Train and evaluate a single model (PyTorch refactor).
    """
    print(f"\n[{model_class.__name__}] Running experiment for embedding dimension {embedding_dimension} with k={k} and hyperparameters: {hyperparameters}")
    td = to_tensor(training_data)
    vd = to_tensor(validation_data)
    tsd = to_tensor(test_data)
    model = model_class(m=getattr(td, 'm', td.shape[0]), n=getattr(td, 'n', td.shape[1]), k=embedding_dimension, device=device)
    if is_pytorch_model(model) and device is not None:
        td = td.to(device) if hasattr(td, 'to') else td
        vd = vd.to(device) if hasattr(vd, 'to') else vd
        tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
        model = model.to(device)
        print(f"Model and data moved to device: {device}")
    else:
        if device is not None and str(device) == 'cuda':
            print(f"Warning: {model_class.__name__} does not support GPU. Running on CPU.")
    tie = TechniqueInferenceEngine(
        training_data=td,
        validation_data=vd,
        test_data=tsd,
        model=model,
        prediction_method=method,
        enterprise_attack_filepath=enterprise_attack_filepath,
    )
    print(f"[{model_class.__name__}] Starting training...")
    mse = tie.fit(**hyperparameters)
    print(f"[{model_class.__name__}] Starting training...")
    print(f"[{model_class.__name__}] Training complete. MSE: {mse}")
    precision = tie.precision(k=k)
    recall = tie.recall(k=k)
    ndcg = tie.normalized_discounted_cumulative_gain(k=k)
    print(f"[{model_class.__name__}] Precision@{k}: {precision}, Recall@{k}: {recall}, NDCG@{k}: {ndcg}")
    metrics = {
        "label": label or model_class.__name__,
        "mse": mse,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
    }
    return tie, metrics


def compare_models(training_data, validation_data, test_data, enterprise_attack_filepath, return_best=False, device=None):
    """
    Train and compare all supported models. Returns best model if return_best=True.
    """
    oilrig_techniques = {
        "T1047", "T1059.005", "T1124", "T1082",
        "T1497.001", "T1053.005", "T1027", "T1105",
        "T1070.004", "T1059.003", "T1071.001"
    }
    k = 20
    embedding_dimension = 10
    model_configs = [
        (TopItemsRecommender, PredictionMethod.DOT, {'gravity_coefficient': 0.001, 'regularization_coefficient': 0.5, 'epochs': 1000, 'learning_rate': 100.0}, "TopItemsRecommender"),
        (FactorizationRecommender, PredictionMethod.DOT, {'gravity_coefficient': 0.001, 'regularization_coefficient': 0.001, 'epochs': 10, 'learning_rate': 1.0}, "FactorizationRecommender"),
        (BPRRecommender, PredictionMethod.COSINE, {'regularization_coefficient': 0.01, 'epochs': 25, 'learning_rate': 0.001}, "BPRRecommender"),
        (ImplicitBPRRecommender, PredictionMethod.COSINE, {'regularization_coefficient': 0.0001, "epochs": 20, 'learning_rate': 0.005}, "ImplicitBPRRecommender"),
        (ImplicitWalsRecommender, PredictionMethod.COSINE, {'regularization_coefficient': 0.05, 'c': 0.5, 'epochs': 20}, "ImplicitWalsRecommender"),
        (WalsRecommender, PredictionMethod.DOT, {'regularization_coefficient': 0.00001, 'c': 0.001, "epochs": 25}, "WalsRecommender"),
    ]
    results = []
    ties = []
    for model_class, method, hyperparams, label in model_configs:
        tie, metrics = run_experiment(
            model_class, method, training_data, validation_data, test_data,
            enterprise_attack_filepath, embedding_dimension, k, hyperparams, label, device=device
        )
        print(f"\nModel: {label}")
        print(metrics)
        if hasattr(tie, "predict_for_new_report"):
            preds = tie.predict_for_new_report(oilrig_techniques, **hyperparams)
            print(preds)
        results.append(metrics)
        ties.append((tie, metrics))
    best = max(results, key=lambda x: x['ndcg'])
    best_idx = results.index(best)
    print(f"\nBest model: {best['label']} (NDCG: {best['ndcg']})")
    best_tie, best_metrics = ties[best_idx]
    save_model(best_tie._model, best_metrics['label'], best_metrics['label'])
    if return_best:
        return ties[best_idx][0], best
    return results


def experiment_wals(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """Run WALS embedding dimension experiment (PyTorch refactor)."""
    print("\n[WALS] Starting WALS embedding dimension experiment...")
    error_count = 0
    results = []
    try:
        sweep_results = test_multiple_embedding_dimensions(
            model_class=WalsRecommender,
            method=PredictionMethod.DOT,
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            enterprise_attack_filepath=enterprise_attack_filepath,
            out_file="wals_model_results_training_data_correction_dot.csv",
            device=device,
            epochs=[25],
            c=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7],
            regularization_coefficient=[0.01, 0.05, 0.1, 0.3, 0.5]
        )
        for res in sweep_results:
            if 'error' in res:
                error_count += 1
            results.append(res)
            if error_count > 5:
                print("[WALS] Aborting experiment: more than five errors encountered.")
                break
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[WALS] Experiment aborted due to error: {e}")
    print("[WALS] Experiment complete. Results:")
    print(results)
    # Save best model by ndcg_at_20 if available
    valid_results = [r for r in results if 'ndcg_at_20' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get('ndcg_at_20', 0))
        print(f"[WALS] Best model by NDCG@20: {best}")
        save_model(WalsRecommender, 'wals', 'WalsRecommender')
    return results


def experiment_bpr(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """Run BPR embedding dimension experiment (PyTorch refactor)."""
    print("\n[BPR] Starting BPR embedding dimension experiment...")
    error_count = 0
    results = []
    try:
        sweep_results = test_multiple_embedding_dimensions(
            model_class=BPRRecommender,
            method=PredictionMethod.DOT,
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            enterprise_attack_filepath=enterprise_attack_filepath,
            out_file="bpr_model_results.csv",
            device=device,
            epochs=[20],
            learning_rate=[0.00001, 0.00005, 0.0001, 0.001],
            regularization=[0., 0.0001, 0.001, 0.01],
        )
        for res in sweep_results:
            if 'error' in res:
                error_count += 1
            results.append(res)
            if error_count > 5:
                print("[BPR] Aborting experiment: more than five errors encountered.")
                break
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[BPR] Experiment aborted due to error: {e}")
    print("[BPR] Experiment complete. Results:")
    print(results)
    valid_results = [r for r in results if 'ndcg_at_20' in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get('ndcg_at_20', 0))
        print(f"[BPR] Best model by NDCG@20: {best}")
        save_model(BPRRecommender, 'bpr', 'BPRRecommender')
    return results


def experiment_topitems(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """
    Run Top Items Recommender experiment (PyTorch refactor). 
    The "Top Items" Recommender is a (naive) model that recommends Techniques in order of their frequency in the dataset.

    This model acts as a baseline and allows us to compare how other models perform against simply guessing the most popular ATT&CK Techniques.
    """
    print("\n[TopItems] Starting TopItems embedding dimension experiment...")
    oilrig_techniques = {
        "T1047", "T1059.005", "T1124", "T1082",
        "T1497.001", "T1053.005", "T1027", "T1105",
        "T1070.004", "T1059.003", "T1071.001"
    }
    embedding_dimension = 10
    k = 20
    best_hyperparameters = {'gravity_coefficient': 0.001, 'regularization_coefficient': 0.5, 'epochs': 1000, 'learning_rate': 100.0}
    error_count = 0
    results = []
    try:
        td = to_tensor(training_data)
        vd = to_tensor(validation_data)
        tsd = to_tensor(test_data)
        model = TopItemsRecommender(m=getattr(td, 'm', td.shape[0]), n=getattr(td, 'n', td.shape[1]), k=embedding_dimension)
        if is_pytorch_model(model) and device is not None:
            td = td.to(device) if hasattr(td, 'to') else td
            vd = vd.to(device) if hasattr(vd, 'to') else vd
            tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
            model = model.to(device)
            print(f"Model and data moved to device: {device}")
        else:
            if device is not None and str(device) == 'cuda':
                print(f"Warning: TopItemsRecommender does not support GPU. Running on CPU.")
        tie = TechniqueInferenceEngine(
            training_data=td,
            validation_data=vd,
            test_data=tsd,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=enterprise_attack_filepath,
        )
        print(f"[TopItems] Starting training...")
        try:
            mse = tie.fit()
            print(f"[TopItems] Training complete. MSE: {mse}")
            precision = tie.precision(k=k)
            recall = tie.recall(k=k)
            ndcg = tie.normalized_discounted_cumulative_gain(k=k)
            print(f"[TopItems] Precision@{k}: {precision}, Recall@{k}: {recall}, NDCG@{k}: {ndcg}")
            new_report_predictions = tie.predict_for_new_report(oilrig_techniques, **best_hyperparameters)
            print("[TopItems] New report predictions:")
            print(new_report_predictions)
            results.append({
                "mse": mse,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "status": "success"
            })
            save_model(model, 'top', 'TopItemsRecommender')
        except Exception as e:
            error_count += 1
            print(f"[TopItems] Error during training: {e}")
            results.append({"error": str(e), "status": "error"})
            if error_count > 5:
                print("[TopItems] Aborting experiment: more than five errors encountered.")
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[TopItems] Experiment aborted due to error: {e}")
    return results


def experiment_factorization(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """
    This Technique Inference Engine uses a Matrix Factorization model to recommend Techniques.

    Use .fit() if you want to explicitly set the model's hyperparameters. Use .fit_with_validation() if you want TIE to choose the best hyperparameters for you.
    """
    print("\n[Factorization Recommender] Starting Factorization embedding dimension experiment...")
    oilrig_techniques = {
        "T1047", "T1059.005", "T1124", "T1082",
        "T1497.001", "T1053.005", "T1027", "T1105",
        "T1070.004", "T1059.003", "T1071.001"
    }
    embedding_dimension = 10
    k = 20
    best_hyperparameters = {'gravity_coefficient': 0.001, 'regularization_coefficient': 0.001, 'epochs': 10, 'learning_rate': 1.0}
    error_count = 0
    results = []
    try:
        td = to_tensor(training_data)
        vd = to_tensor(validation_data)
        tsd = to_tensor(test_data)
        model = FactorizationRecommender(m=training_data.m, n=training_data.n, k=embedding_dimension)
        if is_pytorch_model(model) and device is not None:
            td = td.to(device) if hasattr(td, 'to') else td
            vd = vd.to(device) if hasattr(vd, 'to') else vd
            tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
            model = model.to(device)
            print(f"Model and data moved to device: {device}")
        else:
            if device is not None and str(device) == 'cuda':
                print(f"Warning: TopItemsRecommender does not support GPU. Running on CPU.")
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=enterprise_attack_filepath,
        )
        print(f"[Factorization] Starting training...")
        try:
            mse = tie.fit()
            print(f"[Factorization] Training complete. MSE: {mse}")
            precision = tie.precision(k=k)
            recall = tie.recall(k=k)
            ndcg = tie.normalized_discounted_cumulative_gain(k=k)
            print(f"[Factorization] Precision@{k}: {precision}, Recall@{k}: {recall}, NDCG@{k}: {ndcg}")
            print("[Factorization] New report predictions not supported")
            results.append({
                "mse": mse,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "status": "success"
            })
            save_model(model, 'fact', 'factorization-rec')
        except Exception as e:
            error_count += 1
            print(f"[Factorization] Error during training: {e}")
            results.append({"error": str(e), "status": "error"})
            if error_count > 5:
                print("[Factorization] Aborting experiment: more than five errors encountered.")
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[Factorization] Experiment aborted due to error: {e}")
    return results


def experiment_implicitWALS(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """
    This Technique Inference Engine uses a WALS Recommender model to recommend Techniques.

    This implementation uses the implicit library's implementation of ALS.

    Use .fit() if you want to explicitly set the model's hyperparameters.

    Use .fit_with_validation() if you want TIE to choose the best hyperparameters for you.

    """
    print("[WARNING] ImplicitWalsRecommender uses the implicit library's ALS, which is CPU-only. This is a bottleneck for large datasets. Consider migrating to a GPU-compatible ALS implementation for faster training.")
    # hyperparameters
    embedding_dimension = 10
    k = 20
    best_hyperparameters = {'regularization_coefficient': 0.05, 'c': 0.5, 'epochs': 20}
    error_count = 0
    results = []
    try:
        td = to_tensor(training_data)
        vd = to_tensor(validation_data)
        tsd = to_tensor(test_data)
        model = ImplicitWalsRecommender(m=training_data.m, n=training_data.n, k=embedding_dimension, device=device)
        if hasattr(model, 'to') and device is not None:
            td = td.to(device) if hasattr(td, 'to') else td
            vd = vd.to(device) if hasattr(vd, 'to') else vd
            tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
            model = model.to(device)
            print(f"Model and data moved to device: {device}")
        else:
            if device is not None and str(device) == 'cuda':
                print(f"Warning: ImplicitWALS Recommender does not support GPU. Running on CPU.")
        tie = TechniqueInferenceEngine(
                    training_data=training_data,
                    validation_data=validation_data,
                    test_data=test_data,
                    model=model,
                    prediction_method=PredictionMethod.COSINE,
                    enterprise_attack_filepath=enterprise_attack_filepath)
        try:
            mse = tie.fit(**best_hyperparameters)
            print(f"[ImplicitWALS] Training complete. MSE: {mse}")
            precision = tie.precision(k=k)
            recall = tie.recall(k=k)
            ndcg = tie.normalized_discounted_cumulative_gain(k=k)
            print(f"[ImplicitWALS] Precision@{k}: {precision}, Recall@{k}: {recall}, NDCG@{k}: {ndcg}")
            new_report_predictions = tie.predict_for_new_report(oilrig_techniques, **best_hyperparameters)
            print("[ImplicitWALS] New report predictions:")
            print(new_report_predictions)
            results.append({
                "mse": mse,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "status": "success"
            })
            save_model(model, 'dot', 'ImplicitWALS')
        except Exception as e:
            error_count += 1
            print(f"[ImplicitWALS] Error during training: {e}")
            results.append({"error": str(e), "status": "error"})
            if error_count > 5:
                print("[ImplicitWALS] Aborting experiment: more than five errors encountered.")
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[ImplicitWALS] Experiment aborted due to error: {e}")
    return results


def experiment_implicitBPR(training_data, validation_data, test_data, enterprise_attack_filepath, device=None):
    """
    This Technique Inference Engine also uses a BPR Recommender model to recommend Techniques.

    This implementation uses the implicit library's implementation of BPR.

    Use .fit() if you want to explicitly set the model's hyperparameters.

    Use .fit_with_validation() if you want TIE to choose the best hyperparameters for you.

    Limitations
    Due to limitations in the implicit library, this model does not implement predict_for_new_report().
    """
    # hyperparameters
    embedding_dimension = 10
    k = 20
    best_hyperparameters = {'regularization_coefficient': 0.0001, "epochs": 20, 'learning_rate': 0.005}
    error_count = 0
    results = []
    try:
        td = to_tensor(training_data)
        vd = to_tensor(validation_data)
        tsd = to_tensor(test_data)
        model = ImplicitBPRRecommender(m=training_data.m, n=training_data.n, k=embedding_dimension, device=device)
        if hasattr(model, 'to') and device is not None:
            td = td.to(device) if hasattr(td, 'to') else td
            vd = vd.to(device) if hasattr(vd, 'to') else vd
            tsd = tsd.to(device) if hasattr(tsd, 'to') else tsd
            model = model.to(device)
            print(f"Model and data moved to device: {device}")
        else:
            if device is not None and str(device) == 'cuda':
                print(f"Warning: ImplicitBPR Recommender does not support GPU. Running on CPU.")
        tie = TechniqueInferenceEngine(
                    training_data=training_data,
                    validation_data=validation_data,
                    test_data=test_data,
                    model=model,
                    prediction_method=PredictionMethod.COSINE,
                    enterprise_attack_filepath=enterprise_attack_filepath)
        try:
            mse = tie.fit(**best_hyperparameters)
            print(f"[ImplicitBPR] Training complete. MSE: {mse}")
            precision = tie.precision(k=k)
            recall = tie.recall(k=k)
            ndcg = tie.normalized_discounted_cumulative_gain(k=k)
            print(f"[ImplicitBPR] Precision@{k}: {precision}, Recall@{k}: {recall}, NDCG@{k}: {ndcg}")
            print("[ImplicitBPR] New report predictions not supported")
            results.append({
                "mse": mse,
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg,
                "status": "success"
            })
            save_model(model, 'i_bpr', 'ImplicitBPR')
        except Exception as e:
            error_count += 1
            print(f"[ImplicitBPR] Error during training: {e}")
            results.append({"error": str(e), "status": "error"})
            if error_count > 5:
                print("[ImplicitBPR] Aborting experiment: more than five errors encountered.")
    except Exception as e:
        results.append({"error": str(e), "status": "experiment_aborted"})
        print(f"[ImplicitBPR] Experiment aborted due to error: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Technique Inference Engine Experiments")
    parser.add_argument('--experiment', type=str, choices=[
        'wals', 'bpr', 'topitems', 'factor', 'implicit_bpr', 'implicit_wals', 'compare'
    ], default='compare', help='Experiment to run')
    parser.add_argument('--return_best', action='store_true', help='Return best model')
    args = parser.parse_args()

    # Timestamp for all metrics
    execution_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = get_device()
    training_data, test_data, validation_data, enterprise_attack_filepath = load_data(device)

    if args.experiment == 'wals':
        result = experiment_wals(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "WalsRecommender", execution_timestamp)
    elif args.experiment == 'bpr':
        result = experiment_bpr(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "BPRRecommender", execution_timestamp)
    elif args.experiment == 'factor':
        result = experiment_factorization(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "FactorizationRecommender", execution_timestamp)
    elif args.experiment == 'topitems':
        result = experiment_topitems(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "TopItemsRecommender", execution_timestamp)
    elif args.experiment == 'implicit_bpr':
        result = experiment_implicitBPR(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "ImplicitBPR", execution_timestamp)
    elif args.experiment == 'implicit_wals':
        result = experiment_implicitWALS(training_data, validation_data, test_data, enterprise_attack_filepath, device=device)
        save_metrics_json(result, "ImplicitWALS", execution_timestamp)
    elif args.experiment == 'compare':
        results = compare_models(training_data, validation_data, test_data, enterprise_attack_filepath, return_best=args.return_best, device=device)
         # Save each model's metrics
        for metrics in results:
            save_metrics_json(metrics, metrics.get('label', 'UnknownModel'), execution_timestamp)
        # Save all results as one JSON
        os.makedirs("tie_model", exist_ok=True)
        all_results_path = os.path.join("tie_model", f"all_model_results_{execution_timestamp}.json")
        # Convert all numpy types to native Python types for JSON serialization
        results_converted = convert_types(results)
        with open(all_results_path, "w") as f:
            json.dump(results_converted, f, indent=2)
        print(f"All model results saved to {all_results_path}")

if __name__ == "__main__":
    main()