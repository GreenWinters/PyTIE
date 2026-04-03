# PyTIE

## Technique Inference Engine (TIE): Training Methods and Implementation Details

The Technique Inference Engine (TIE) is a machine learning system that enables cyber defenders to forecast an adversary's next steps by predicting associated [MITRE ATT&CK](https://attack.mitre.org/) techniques from previously observed behaviors. TIE leverages one of the largest publicly available datasets linking CTI Reports to ATT&CK Techniques, with 43,899 technique observations across 6,236 CTI Reports, covering 96% of ATT&CK Enterprise v15.0.

### Functionality Summary

The core functionality of `tie_model.py` is to orchestrate the training, evaluation, and comparison of multiple recommender models for technique inference. It provides a CLI for experiment selection, automates device management (CPU/GPU), hyperparameter sweeps, and outputs trained model artifacts and detailed metrics. This supports TIE’s goal of enabling defenders to build a complete picture of adversary actions and forecast future techniques.

### Training Methodology

#### Inputs
- **Dataset:**
	- `combined_dataset_full_frequency.json`: Main report-technique matrix (43,899 observations, 6,236 reports).
	- `enterprise-attack.json`: MITRE ATT&CK technique mapping.
- **Structure:**
	- Data is split into training, validation, and test sets using `ReportTechniqueMatrixBuilder`.
	- All splits are converted to PyTorch tensors for efficient computation and device compatibility.

#### Model Training
- **Supported Models:**
	- WALS (Weighted Alternating Least Squares)
		- **Description:** Matrix factorization algorithm that alternates between optimizing user and item embeddings using weighted least squares. Designed for implicit feedback data, it efficiently handles large, sparse matrices.
		- **Strengths:** Scales well to large datasets, robust to missing data, supports GPU acceleration, interpretable embeddings.
		- **Weaknesses:** Assumes linear relationships, may struggle with highly non-linear or complex interactions, sensitive to hyperparameters.
		- **Prediction Target:** Predicts the likelihood of an adversary using a specific ATT&CK technique based on historical report-technique associations.
		- **Output Interpretation:** Higher scores indicate greater predicted relevance; top-k techniques are recommended for further investigation.
	- BPR (Bayesian Personalized Ranking)
		- **Description:** Pairwise ranking algorithm that learns to rank observed (positive) interactions higher than unobserved (negative) ones. Optimizes for ranking quality rather than absolute prediction.
		- **Strengths:** Directly optimizes ranking metrics, effective for recommendation tasks, handles implicit feedback.
		- **Weaknesses:** May require careful negative sampling, slower convergence on very large datasets, less interpretable than WALS.
		- **Prediction Target:** Ranks ATT&CK techniques for each report/entity by predicted relevance.
		- **Output Interpretation:** Techniques with highest predicted rank are most likely to be used by the adversary.
	- TopItems (Frequency-based Baseline)
		- **Description:** Simple baseline that recommends techniques based on their overall frequency in the dataset, ignoring report-specific context.
		- **Strengths:** Fast, easy to implement, provides a baseline for comparison.
		- **Weaknesses:** Ignores context and personalization, may miss rare but important techniques.
		- **Prediction Target:** Most frequently observed ATT&CK techniques.
		- **Output Interpretation:** Top-k most common techniques are recommended.
	- Factorization (Matrix Factorization)
		- **Description:** Learns latent embeddings for reports and techniques by factorizing the report-technique matrix. Captures underlying structure and associations.
		- **Strengths:** Captures latent structure, interpretable embeddings, efficient for moderate-sized datasets.
		- **Weaknesses:** Assumes linear relationships, less effective for highly non-linear data, sensitive to sparsity.
		- **Prediction Target:** Predicts technique relevance for each report/entity.
		- **Output Interpretation:** Higher scores indicate stronger predicted association; top-k techniques are recommended.
	- Implicit BPR (via implicit.gpu)
		- **Description:** GPU-accelerated Bayesian Personalized Ranking implemented through the `implicit.gpu.bpr` backend. Trains directly on CUDA tensors and runs efficiently with the conda-forge GPU build (`conda install -c conda-forge implicit implicit-proc=*=gpu`).
		- **Strengths:** Highly scalable ranking, runs on GPU, robust to missing data.
		- **Weaknesses:** Requires careful tuning, less interpretable than standard matrix factorization, sensitive to data distribution.
		- **Prediction Target:** Ranks ATT&CK techniques for each report/entity.
		- **Output Interpretation:** Techniques with highest predicted rank are most likely to be used by the adversary.
	- Implicit WALS (deprecated CPU wrapper)
		- **Description:** Compatibility shim for the CPU-only `implicit.als` implementation. Maintained for backwards compatibility but emits a deprecation warning and should be avoided when GPU acceleration is required.
		- **Strengths:** Matches legacy behavior.
		- **Weaknesses:** CPU-only, significantly slower, no GPU acceleration, less future proof.
	- Implicit ALS (GPU/CPU fallback)
		- **Description:** Alternating Least Squares recommender using `implicit.gpu.als` when CUDA extensions are available, with automatic fallback to CPU ALS.
		- **Strengths:** Efficient on sparse implicit-feedback data, supports GPU acceleration, robust fallback behavior.
		- **Weaknesses:** Backend/library compatibility can impact runtime, and CPU fallback is substantially slower on large runs.
		- **Prediction Target:** Predicts and ranks ATT&CK technique relevance for each report/entity from learned latent factors.
		- **Output Interpretation:** Higher scores indicate stronger predicted relevance; top-k techniques are prioritized.
	- Implicit MatrixFactorizationBase (GPU/CPU fallback)
		- **Description:** Adapter around `implicit` matrix factorization base models, preferring GPU implementations when available and falling back to CPU when needed.
		- **Strengths:** Flexible backend selection, consistent recommender interface, resilient to missing CUDA extensions.
		- **Weaknesses:** Performance depends on backend and environment setup; fallback paths can increase runtime.
		- **Prediction Target:** Learns user/report and item/technique latent representations to score likely technique associations.
		- **Output Interpretation:** Scores are used for ranking; highest-ranked techniques are recommended as likely next actions.
- **Device Management:**
	- TIE detects CUDA availability at runtime and chooses the execution device automatically.
	- For PyTorch-based recommenders, model parameters and tensors are explicitly moved to the selected device.
	- For `implicit`-based recommenders, GPU backends are used when available and compatible; otherwise, code falls back to CPU implementations.
	- Device-aware training means the same experiment interface runs across heterogeneous environments while preserving comparable outputs/metrics.
- **Hyperparameter Sweeps:**
	- Automated sweeps over embedding dimensions and hyperparameters, with validation-based selection of best configurations.
- **Training Loop:**
	- For each configuration, the model is trained, validated, and tested. Best hyperparameters are selected based on performance metrics.

#### Outputs
- **Model Artifacts:**
	- Trained models saved as PyTorch `.pt` files with timestamped filenames.
- **Metrics:**
	- Performance metrics saved as JSON files, including precision, recall, NDCG, and MSE.
- **CSV Results:**
	- Embedding dimension sweeps and other experiments output results as CSV files.

#### Measurements/Metrics
- **Precision@k:** Proportion of relevant techniques among top-k predictions.
- **Recall@k:** Proportion of relevant techniques retrieved among all possible relevant techniques.
- **NDCG@k:** Evaluates ranking quality, rewarding correct ordering of relevant techniques.
- **MSE:** Used for regression-based models to assess fit quality.
- **Best Model Selection:** Models compared and best selected by NDCG@20.

### Distinction from Original Repository

Compared to the original MITRE Engenuity Center for Threat-Informed Defense repository, this version of `tie_model.py` includes:
- **Device-aware training:** All models and tensors are moved to GPU if available.
- **Expanded model support:** Additional recommender models (Implicit BPR/WALS) integrated.
- **Automated hyperparameter sweeps:** Multiple configurations run in parallel, best selected by validation metrics.
- **Robust output management:** Model artifacts and metrics saved with timestamped filenames.
- **Comprehensive CLI:** Experiment selection, best model return, and output in JSON/CSV formats.
- **Enhanced metrics and reporting:** Detailed metrics for each experiment, all results saved for analysis.
- **Integration with RL environments:** Output artifacts and metrics designed for downstream RL and simulation pipelines.
- **Error handling and progress feedback:** Robust error handling, progress print statements, and device compatibility checks.


## Poetry File: Function and Purpose

This repository uses a `pyproject.toml` file managed by [Poetry](https://python-poetry.org/) for dependency management and packaging. Poetry simplifies installation, version control, and reproducibility of Python environments, ensuring that all required libraries and their versions are tracked and easily installed.

For reproducible GPU workflows, this project also includes `environment.yml` (Conda) to define the CUDA-aligned base environment used by the experiments.

Recommended setup sequence:

1. Create and activate the Conda environment from `environment.yml`:
	```sh
	conda env create -f technique_inference_engine/environment.yml
	conda activate tie_gpu
	```
2. Install project dependencies with Poetry:
	```sh
	poetry install
	```
3. Ensure GPU-enabled `implicit` artifacts are installed (if not already present in the active env):
	```sh
	conda install -c conda-forge implicit implicit-proc=*=gpu
	```

In short: `environment.yml` establishes the system-level Python/CUDA baseline, while Poetry (`pyproject.toml`) manages project-level Python package versions and lockfile reproducibility.

---

## System Reference
This repository was altered and tested on a multi-GPU workstation with the following configuration:
- GPUs (4): NVIDIA RTX A6000, NVIDIA GeForce RTX 2080 Ti
- OS: Ubuntu 22.04.5 LTS
- CPU Model: Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz
- CPU(s): 40

This hardware and OS configuration enabled accelerated training and large-scale experiments for the recommender models described above.

---

## License and Attribution

This repository is based on the original open-source release at [https://github.com/center-for-threat-informed-defense/technique-inference-engine/tree/TIE-83-jupyter-instructions](https://github.com/center-for-threat-informed-defense/technique-inference-engine/tree/TIE-83-jupyter-instructions) and incorporates modifications and enhancements for research and development purposes. Please support and cite the original release and its authors.

This project is distributed under the terms of the Apache License, Version 2.0. You may use, reproduce, and distribute this work and derivative works, provided you comply with the conditions of the license, including prominent notices of modification, retention of attribution, and inclusion of the license text. See the LICENSE file and the full license text above for details.

For more information, visit the [Technique Inference Engine Website](https://center-for-threat-informed-defense.github.io/technique-inference-engine/).

---
