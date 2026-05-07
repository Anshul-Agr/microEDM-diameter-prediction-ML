import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import os
import json
import time
import joblib

# --- 0. Global Setup ---
sns.set_style("whitegrid")
OUTPUT_DIR = 'xgboost_multi_output_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open('hyperparameter_tuning_results/xgboost_best_params.json', 'r') as f:
        tuned_params = json.load(f)
    print("Loaded best XGBoost hyperparameters.")
except FileNotFoundError:
    print("Error: xgboost_best_params.json not found. Run the hyperparameter tuning script first.")
    exit()

input_features = ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)']
output_features = ['Entry dia', 'Exit dia', 'MRR', 'Taper(degree)']

# --- 1. Main Experiment Function ---
def run_xgb_experiment(seed: int, df: pd.DataFrame, all_best_params: dict):
    print(f"\n--- Running XGBoost Experiment with Seed: {seed} ---")

    X = df[input_features]
    y = df[output_features]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=seed)

    # --- A. Scaling ---
    scaler_X = StandardScaler()
    scaler_X.set_output(transform="pandas")
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    scaler_y.set_output(transform="pandas")
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # --- B. Model Training (per-target) ---
    best_model_collection = {}
    for feature in output_features:
        feature_params = all_best_params[feature].copy()
        feature_params['random_state'] = seed
        feature_params['n_jobs'] = 1
        feature_params['early_stopping_rounds'] = 50

        model = xgb.XGBRegressor(**feature_params)
        model.fit(
            X_train_scaled, y_train_scaled[feature],
            eval_set=[(X_val_scaled, y_val_scaled[feature])],
            verbose=False
        )
        best_model_collection[feature] = model

    # --- C. Evaluation ---
    test_preds_scaled = {f: m.predict(X_test_scaled) for f, m in best_model_collection.items()}
    y_pred_scaled_df = pd.DataFrame(test_preds_scaled, index=X_test.index)
    y_pred = scaler_y.inverse_transform(y_pred_scaled_df)
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index, columns=output_features)
    y_pred_df[y_pred_df < 0] = 0
    test_r2_scores = {f: r2_score(y_test[f], y_pred_df[f]) for f in output_features}

    train_preds_scaled = {f: m.predict(X_train_scaled) for f, m in best_model_collection.items()}
    train_pred_scaled_df = pd.DataFrame(train_preds_scaled, index=X_train.index)
    train_pred = scaler_y.inverse_transform(train_pred_scaled_df)
    train_pred_df = pd.DataFrame(train_pred, index=y_train.index, columns=output_features)
    train_pred_df[train_pred_df < 0] = 0
    train_r2_scores = {f: r2_score(y_train[f], train_pred_df[f]) for f in output_features}

    full_preds_scaled = {f: m.predict(scaler_X.transform(X)) for f, m in best_model_collection.items()}
    full_pred_scaled_df = pd.DataFrame(full_preds_scaled, index=X.index)
    full_pred = scaler_y.inverse_transform(full_pred_scaled_df)
    full_pred_df = pd.DataFrame(full_pred, index=y.index, columns=output_features)
    full_pred_df[full_pred_df < 0] = 0
    full_r2_scores = {f: r2_score(y[f], full_pred_df[f]) for f in output_features}

    # --- D. Report DataFrames ---
    train_report_df = y_train.rename(columns=lambda c: f'Actual_{c}').join(train_pred_df.rename(columns=lambda c: f'Predicted_{c}'))
    test_report_df = y_test.rename(columns=lambda c: f'Actual_{c}').join(y_pred_df.rename(columns=lambda c: f'Predicted_{c}'))
    full_report_df = y.rename(columns=lambda c: f'Actual_{c}').join(full_pred_df.rename(columns=lambda c: f'Predicted_{c}'))

    print(f"  Test R2 for seed {seed}: {[f'{k}: {v:.4f}' for k, v in test_r2_scores.items()]}")

    # Save surrogate bundle for seed 0 (used by Surrogate.py)
    if seed == 0:
        model_save_path = os.path.join(OUTPUT_DIR, 'xgboost_model_collection_seed0.joblib')
        joblib.dump(best_model_collection, model_save_path)
        surrogate_bundle = {
            "models": best_model_collection,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "input_features": input_features,
            "output_features": output_features,
        }
        bundle_path = os.path.join(OUTPUT_DIR, "xgboost_surrogate_seed0.joblib")
        joblib.dump(surrogate_bundle, bundle_path)
        print(f"  Surrogate bundle saved to: {bundle_path}")

    return test_r2_scores, train_r2_scores, full_r2_scores, train_report_df, test_report_df, full_report_df

# --- 2. Data Loading and Preparation ---
try:
    df_main = pd.read_csv('Final.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Final.csv' not found.")
    exit()

columns_to_drop = ['Sl No. (Image)', 'Unnamed: 6', 'OC at Entry', 'OC at Exit']
df_main = df_main.drop(columns=columns_to_drop, errors='ignore')

# --- 3. Execute the 50-Run Experiment ---
print("\n--- Starting 50-Run XGBoost Evaluation ---")
start_eval_time = time.time()
num_runs = 50
all_test_r2, all_train_r2, all_full_r2 = [{f: [] for f in output_features} for _ in range(3)]
all_train_reports, all_full_reports, all_test_reports = [], [], []

for i in range(num_runs):
    test_scores, train_scores, full_scores, train_report, test_report, full_report = run_xgb_experiment(seed=i, df=df_main, all_best_params=tuned_params)
    for feature in output_features:
        all_test_r2[feature].append(test_scores[feature])
        all_train_r2[feature].append(train_scores[feature])
        all_full_r2[feature].append(full_scores[feature])
    train_report['run'] = i
    test_report['run'] = i
    full_report['run'] = i
    all_train_reports.append(train_report)
    all_test_reports.append(test_report)
    all_full_reports.append(full_report)

total_eval_time = time.time() - start_eval_time
print(f"\n--- Evaluation Complete ---")
print(f"  Total time for {num_runs} runs: {total_eval_time:.2f} seconds")

# --- 4. Analyze and Report Final Results ---
print(f"\n--- Final Aggregated XGBoost Results from {num_runs} Runs ---")
print("\nTest SET:")
for feature, scores in all_test_r2.items():
    print(f"  {feature}: R2 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print("\nTraining SET (Diagnostics):")
for feature, scores in all_train_r2.items():
    print(f"  {feature}: R2 = {np.mean(scores):.4f}")
print("\nFull DATASET (Diagnostics):")
for feature, scores in all_full_r2.items():
    print(f"  {feature}: R2 = {np.mean(scores):.4f}")

# --- 5. Generate and Save CSV Reports ---
testing_report_all_runs = pd.concat(all_test_reports)
testing_csv_path = os.path.join(OUTPUT_DIR, 'xgboost_testing_predictions_all_runs.csv')
testing_report_all_runs.to_csv(testing_csv_path, index=True)
print(f"Testing predictions saved to: {testing_csv_path}")

training_report_all_runs = pd.concat(all_train_reports)
training_csv_path = os.path.join(OUTPUT_DIR, 'xgboost_training_predictions_all_runs.csv')
training_report_all_runs.to_csv(training_csv_path, index=True)
print(f"Training predictions saved to: {training_csv_path}")

full_report_all_runs = pd.concat(all_full_reports)
full_csv_path = os.path.join(OUTPUT_DIR, 'xgboost_full_dataset_predictions_all_runs.csv')
full_report_all_runs.to_csv(full_csv_path, index=True)
print(f"Full dataset predictions saved to: {full_csv_path}")

# --- 6. Save Raw R-squared Scores ---
output_dir_2 = 'r2_analysis_results'
os.makedirs(output_dir_2, exist_ok=True)
raw_scores_filename = os.path.join(output_dir_2, 'raw_r2_scores_xgboost.json')
with open(raw_scores_filename, 'w') as f:
    json.dump(all_test_r2, f, indent=4)
print(f"Raw R2 scores saved to: {raw_scores_filename}")

# --- 7. Per-Seed R2 with Confidence Intervals ---
from scipy import stats

def summarize(samples):
    vals = np.asarray(samples, dtype=float)
    n = len(vals)
    mean = float(np.mean(vals)) if n else np.nan
    std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n-1)
        ci = (mean - tcrit * se, mean + tcrit * se)
    else:
        ci = (np.nan, np.nan)
    return mean, std, ci

def per_seed_r2(df, actual_col, pred_col):
    rows = []
    for seed, g in df.groupby('run'):
        rows.append((seed, r2_score(g[actual_col], g[pred_col])))
    return pd.DataFrame(rows, columns=['run', 'r2']).sort_values('run', ignore_index=True)

df_test = pd.read_csv(testing_csv_path, index_col=0)
r2_entry_test = per_seed_r2(df_test, 'Actual_Entry dia', 'Predicted_Entry dia')
r2_exit_test  = per_seed_r2(df_test, 'Actual_Exit dia',  'Predicted_Exit dia')
mean_e_t, std_e_t, ci_e_t = summarize(r2_entry_test['r2'])
mean_x_t, std_x_t, ci_x_t = summarize(r2_exit_test['r2'])
print("\nXGBoost -- TEST 20% R2 across seeds (mean +/- std, 95% CI)")
print(f"Entry dia: mean={mean_e_t:.4f}, std={std_e_t:.4f}, 95% CI=({ci_e_t[0]:.4f}, {ci_e_t[1]:.4f})")
print(f"Exit dia : mean={mean_x_t:.4f}, std={std_x_t:.4f}, 95% CI=({ci_x_t[0]:.4f}, {ci_x_t[1]:.4f})")
r2_entry_test.to_csv('r2_analysis_results/xgb_test_r2_per_seed_entry.csv', index=False)
r2_exit_test.to_csv('r2_analysis_results/xgb_test_r2_per_seed_exit.csv', index=False)

df_full = pd.read_csv(full_csv_path, index_col=0)
rows_entry, rows_exit = [], []
for seed, g in df_full.groupby('run'):
    rows_entry.append((seed, r2_score(g['Actual_Entry dia'], g['Predicted_Entry dia'])))
    rows_exit.append((seed, r2_score(g['Actual_Exit dia'],  g['Predicted_Exit dia'])))
r2_entry_full_seeds = pd.DataFrame(rows_entry, columns=['run','r2']).sort_values('run', ignore_index=True)
r2_exit_full_seeds  = pd.DataFrame(rows_exit,  columns=['run','r2']).sort_values('run', ignore_index=True)
mean_e_f, std_e_f, ci_e_f = summarize(r2_entry_full_seeds['r2'])
mean_x_f, std_x_f, ci_x_f = summarize(r2_exit_full_seeds['r2'])
print("\nXGBoost -- FULL 112 R2 across seeds (mean +/- std, 95% CI)")
print(f"Entry dia: mean={mean_e_f:.4f}, std={std_e_f:.4f}, 95% CI=({ci_e_f[0]:.4f}, {ci_e_f[1]:.4f})")
print(f"Exit dia : mean={mean_x_f:.4f}, std={std_x_f:.4f}, 95% CI=({ci_x_f[0]:.4f}, {ci_x_f[1]:.4f})")
r2_entry_full_seeds.to_csv('r2_analysis_results/xgb_full_r2_per_seed_entry.csv', index=False)
r2_exit_full_seeds.to_csv('r2_analysis_results/xgb_full_r2_per_seed_exit.csv', index=False)
