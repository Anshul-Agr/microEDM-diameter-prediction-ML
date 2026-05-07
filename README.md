# Micro-EDM Hole Geometry Prediction

Machine learning pipeline for predicting micro-hole geometry (entry diameter, exit diameter, MRR, taper) in micro-EDM drilling. Compares five regression models across 50 random seeds with statistical aggregation.

## Dataset

Place `Final.csv` in the project root. The dataset contains experimental micro-EDM measurements with the following input parameters:
- Capacitance (pF)
- Voltage (V)
- TRS (rpm)
- Feed rate (µm/s)

## Models

| Script | Model | Hyperparameter Script |
|---|---|---|
| `xgboost.py` | XGBoost (per-target) | `xgboost_hyperparam.py` |
| `lightgbm.py` | LightGBM (per-target) | `lightgbm_hyperparam.py` |
| `SVR.py` | Support Vector Regression | `SVM_hyperparam.py` |
| `MLP.py` | MLP Neural Network | `MLPReg_hyperparam.py` |
| `tensorflow.py` | TensorFlow DNN | `tensorflow_hyperparam.py` |

## Analysis

| Script | Purpose |
|---|---|
| `SHAP.py` | SHAP feature importance analysis (XGBoost-based) |
| `permutation_importance.py` | Permutation importance with statistical testing |
| `Surrogate.py` | Surrogate-based inverse design and taper optimization |

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Hyperparameter tuning (run once per model)
```bash
python xgboost_hyperparam.py
python lightgbm_hyperparam.py
python SVM_hyperparam.py
python MLPReg_hyperparam.py
python tensorflow_hyperparam.py
```

Tuned parameters are saved to `hyperparameter_tuning_results/`.

### 3. Model training and evaluation (50-seed runs)
```bash
python xgboost.py
python lightgbm.py
python SVR.py
python MLP.py
python tensorflow.py
```

Each script outputs:
- Per-seed predictions (CSV)
- Aggregated R² with 95% confidence intervals
- Raw R² scores for statistical comparison

### 4. Feature importance analysis
```bash
python SHAP.py
python permutation_importance.py
```

### 5. Inverse design
```bash
python Surrogate.py
```

Requires the XGBoost surrogate bundle from step 3 (`xgboost_surrogate_seed0.joblib`).

## Output Structure

```
hyperparameter_tuning_results/   # Tuned hyperparameters (JSON)
xgboost_multi_output_model/      # XGBoost predictions and model artifacts
lgbm_multi_output_model/         # LightGBM predictions
svm_multi_output_model/          # SVM predictions
mlp_multi_output_model/          # MLP predictions
tf_multi_output_model/           # TensorFlow predictions
r2_analysis_results/             # Per-seed R² scores and CI analysis
shap_clean/                      # SHAP figures and data exports
feature_importance_results/      # Permutation importance results
INVERSE_plots/                   # Surrogate inverse design plots
INVERSE_results/                 # Inverse design CSV outputs
```
