# bt4012-kaggle-competition
BT4012 Kaggle Competition AY2627 Semester 1
Objective: To build a state-of-the-art fraud detection model that can accurately distinguish between licit and illicit Bitcoin transactions.

---

## Table of Contents
1. [Overview and Highlights](#overview-and-highlights)
2. [Installation and Requirments](#installation-and-requirements)
3. [Final Submission 1 Version 23B](#final-submission-1-version-23b)
4. [Final Submission 2 Version 22](#final-submission-2-version-22)

---

## Overview and Highlights

Detecting financial fraud on the Bitcoin blockchain presents three fundamental challenges:
* **Extreme Class Imbalance:** Confirmed illicit transactions represent ~2.6% of total network volume.
* **Complex Graph Topology:** Laundering transactions are deeply interconnected across multi-hop payment chains.
* **Temporal Dynamics:** Models need to generalise across unseen future time steps without temporal data leakage.

### Key Contributions:
* **Graph Feature Engineering:** Extracts temporal HITS scores (Hub/Authority), degree ratios, shortest path distances to transaction sinks/sources, and predecessor/successor neighborhood aggregations.
* **GATv2 Node Embeddings:** 
    * Version 23B: Trains a 2-layer, multi-head GATv2 trained transductively on all nodes using only the 165 raw features and the edges. It is trained without labels by link prediction 
    * Verion 22: Trains a 2-layer multi-head Graph Attention Network on historical graph topologies (Time steps 1–29) to extract continuous representations capturing topological clustering and label homophily.
* **Hybrid Classification:** Feeds GAT node embeddings and graph features into hyperparameter-tuned **XGBoost** and **LightGBM** ensembles trained using temporal expanding-window cross-validation.

---

## Installation and Requirements

### Prerequisites
* Python 3.10+

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/xinweixw/bt4012-kaggle-competition.git
cd bt4012-kaggle-competition
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate    # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Place the data files in the same folder as the code:**

```
bt4012-kaggle-competition/
├── .venv  # Included in .gitignore
├── .gitignore
├── e1156227_final submission 1_version_23B.ipynb
├── e1156227_final submission 2_version_22.ipynb                           
├── requirements.txt
├── submission_version_22.csv # created upon running e1156227_final submission 2_version_22.ipynb 
├── submission_version_23B.csv # created upon running e1156227_final submission 1_version23B.ipynb
├── test.csv
├── train.csv
├── txs_edgelist.csv
└── README.md
```

---

## Final Submission 1 Version 23B

The notebook (`e1156227_final submission 1_version_23B.ipynb`) contains the code to output the submission file `submision_version_23B.csv`. `submision_version_23B.csv` is my best scoring file on the private leaderboard.

### Project Architecture

```
                  ┌───────────────────────────────────────┐
                  │  BT4012 Bitcoin Train and Edges Data  │
                  └──────────────┬────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      ┌─────────────────────────┐ ┌───────────────────────────┐
      │ Structural Graph        │ │ GATv2 Neural Network      │
      │ Feature Engineering     │ │ Unsupervised Training     │
      │ (Degree, Hits, Distance)│ │ (Transductive Embeddings) │
      │ (Neighbour Means)       │ │                           │
      └────────────┬────────────┘ └────────────┬──────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Feature Set Selection │
                     └───────────┬───────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      ┌─────────────────────────┐ ┌─────────────────────────┐
      │    Optuna-Tuned         │ │    Optuna-Tuned         │
      │      XGBoost            │ │       LightGBM          │
      └────────────┬────────────┘ └────────────┬────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │ (Comparison)
                                 ▼
                    ┌──────────────────────────┐
                    │ Final Fraud Predictions  │
                    │       on Test Data       │
                    └──────────────────────────┘
```
The notebook (`e1156227_final submission 1_version_23B.ipynb`) executes the following sequential steps:

1. **Data Loading & Preprocessing:** Loads `train.csv`, `test.csv`, `txs_edgelist.csv`. Explore dataset data and validate feature integrity
2. **Unsupervised GAT Embeddings:**
   * Constructs a `torch_geometric` PyG Data graph structure.
   * Trains a 2-layer GATv2 model (multi-head attention) transductively on all nodes using only the 165 raw features and the edges. 
   * Extracts 64-dimensional continuous node embeddings.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (10 trials) over GAT hyperparameters.
3. **Graph Feature Engineering:**
   * Splits training data (`train.csv`) into training, validation and hold-out sets by time steps.
   * Computes dynamic `in_deg`, `out_deg`, `deg_total`, `deg_ratio`.
   * Evaluates dynamic HITS scores (`hub_score`, `authority_score`) per time step using `networkx`.
   * Calculates shortest path distances from graph sources and to sinks (`dist_to_sink`, `dist_to_source`).
   * Generates neighborhood feature aggregates (`pred_mean_*`, `succ_mean_*`).
4. **XGBoost Modelling:**
   * **Imbalance Handling Strategy:** Evaluates various imbalance handling strategies (`none`, `class_weights`, `undersampling`, `SMOTE`, `SMOTETomek`)
   * **Feature Selection:** Evaluates feature importance and selects best feature set.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (40 trials) over XGBoost hyperparameters using expanding-window temporal cross-validation.
4. **LightGBM Modelling:**
   * **Imbalance Handling Strategy:** Evaluates various imbalance handling strategies (`none`, `class_weights`, `undersampling`, `SMOTE`, `SMOTETomek`)
   * **Feature Selection:** Evaluates feature importance and selects best feature set.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (40 trials) over LightGBM hyperparameters using expanding-window temporal cross-validation.
6. **Final Evaluation and Submission:** Evaluate XGBoost and LightGBM on the hold-out set. Trains the best model with the selected feature sets and best parameters (from Hyperparameter Tuning) on the entire labelled training data and output the prediction for the test set in `submission_version_23B.csv`.

---

## Final Submission 2 Version 22

The notebook (`ee1156227_final submission 2_version_22.ipynb`) contains the code to output the submission file `submision_version_22.csv`. `submision_version_22.csv` is my second selected output file to be considered on the private leaderboard.

### Project Architecture

```
                  ┌───────────────────────────────────────┐
                  │  BT4012 Bitcoin Train and Edges Data  │
                  └──────────────┬────────────────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      ┌─────────────────────────┐ ┌───────────────────────────┐
      │ Structural Graph        │ │ GATv2 Neural Network      │
      │ Feature Engineering     │ │ (Steps 1–29)              │
      │ (Degree, Hits, Distance)│ │ (Transductive Embeddings) │
      │ (Neighbour Means)       │ │                           │
      └────────────┬────────────┘ └────────────┬──────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Feature Set Selection │
                     └───────────┬───────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
      ┌─────────────────────────┐ ┌─────────────────────────┐
      │    Optuna-Tuned         │ │    Optuna-Tuned         │
      │      XGBoost            │ │       LightGBM          │
      └────────────┬────────────┘ └────────────┬────────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │ (Comparison)
                                 ▼
                    ┌──────────────────────────┐
                    │ Final Fraud Predictions  │
                    │       on Test Data       │
                    └──────────────────────────┘
```
The notebook (`e1156227_final submission 2_version_22.ipynb`) executes the following sequential steps:

1. **Data Loading & Preprocessing:** Loads `train.csv`, `test.csv`, `txs_edgelist.csv`. Explore dataset data and validate feature integrity
2. **Supervised GAT Embeddings:**
   * Constructs a `torch_geometric` PyG Data graph structure.
   * Trains a 2-layer GATv2 model (multi-head attention) with class weighting on time steps 1–29.
   * Extracts 32-dimensional continuous node embeddings.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (30 trials) over GAT hyperparameters.
3. **Graph Feature Engineering:**
   * Splits training data (`train.csv`) into training, validation and hold-out sets by time steps.
   * Computes dynamic `in_deg`, `out_deg`, `deg_total`, `deg_ratio`.
   * Calculates shortest path distances from graph sources and to sinks (`dist_to_sink`, `dist_to_source`).
   * Generates neighborhood feature aggregates (`pred_mean_*`, `succ_mean_*`).
4. **XGBoost Modelling:**
   * **Imbalance Handling Strategy:** Evaluates various imbalance handling strategies (`none`, `class_weights`, `undersampling`, `SMOTE`, `SMOTETomek`)
   * **Feature Selection:** Evaluates feature importance and selects best feature set.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (40 trials) over XGBoost hyperparameters using expanding-window temporal cross-validation.
4. **LightGBM Modelling:**
   * **Imbalance Handling Strategy:** Evaluates various imbalance handling strategies (`none`, `class_weights`, `undersampling`, `SMOTE`, `SMOTETomek`)
   * **Feature Selection:** Evaluates feature importance and selects best feature set.
   * **Hyperparameter Tuning:** Runs **Optuna** Bayesian optimization (40 trials) over LightGBM hyperparameters using expanding-window temporal cross-validation.
6. **Final Evaluation and Submission:** Evaluate XGBoost and LightGBM on the hold-out set. Trains the best model with the selected feature sets and best parameters (from Hyperparameter Tuning) on the entire labelled training data and output the prediction for the test set in `submission_version_22.csv`.