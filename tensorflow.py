import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
import os
import json
import joblib
import time

# --- 0. Global Setup ---
sns.set_style("whitegrid")
OUTPUT_DIR = 'tf_multi_output_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    with open('hyperparameter_tuning_results/tensorflow_best_params.json', 'r') as f:
        tuned_params = json.load(f)
    print("Loaded best TensorFlow hyperparameters.")
except FileNotFoundError:
    print("Error: tensorflow_best_params.json not found. Run the hyperparameter tuning script first.")
    exit()

# Define feature lists globally
INPUT_FEATURES = ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)']
OUTPUT_FEATURES = ['Entry dia', 'Exit dia', 'MRR', 'Taper(degree)']

# --- Custom R² Callback ---
class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_true_scaled = self.validation_data
        y_val_pred_scaled = self.model.predict(X_val, verbose=0)
        r2 = r2_score(y_val_true_scaled, y_val_pred_scaled)
        logs['val_r2'] = r2
# --- Custom Weighted MSE Loss ---
def create_weighted_mse(loss_weights):
    def weighted_mse(y_true, y_pred):
        se = tf.square(y_true - y_pred)
        weighted = se * tf.constant(loss_weights, dtype=se.dtype)
        return tf.reduce_mean(weighted)
    return weighted_mse

# --- 1. Main Experiment Function ---
def run_tf_experiment(seed: int, df: pd.DataFrame, best_hps: dict):
    print(f"\n--- Running TensorFlow Experiment with Seed: {seed} ---")
    
    # --- A. Data Preparation ---
    X = df[INPUT_FEATURES]
    y_raw = df[OUTPUT_FEATURES].copy()
    
    X_train_full, X_test, y_train_full_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X_train_full, y_train_full_raw, test_size=0.2, random_state=seed
    )

    # --- B. Scaling ---
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train_raw)

    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train_raw)
    y_val_scaled = scaler_y.transform(y_val_raw)

    # --- C. Weighted Loss Setup (Targeting Entry/Exit) ---
    tf.random.set_seed(seed)
    
    # Calculate inverse variance for all to start
    target_variances = np.var(y_train_scaled, axis=0)
    loss_weights = np.ones(len(OUTPUT_FEATURES), dtype=np.float32)
    
    # Apply high-priority inverse variance weights to Entry (0) and Exit (1)
    loss_weights[0] = 1.0 / (target_variances[0] + 1e-6)
    loss_weights[1] = 1.0 / (target_variances[1] + 1e-6)
    
    # Set MRR (2) and Taper (3) to low priority
    loss_weights[2] = 0.1
    loss_weights[3] = 0.1
    
    # Normalize weights so mean is 1.0
    loss_weights /= np.mean(loss_weights)

    # --- D. Model Building ---
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
    
    for i in range(best_hps.get('num_layers')):
        model.add(tf.keras.layers.Dense(
            units=best_hps.get(f'units_{i}'),
            activation=best_hps.get('activation')
        ))
    
    model.add(tf.keras.layers.Dense(len(OUTPUT_FEATURES), activation='linear'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
        loss=create_weighted_mse(loss_weights) 
    )

    # --- E. Training ---
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=500, 
        batch_size=X_train_scaled.shape[0], 
        callbacks=[early_stop], 
        verbose=0
    )

    # --- F. Predictions and Inverse Scaling ---
    # Test predictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred_raw = scaler_y.inverse_transform(y_pred_scaled)
    
    y_test_final = y_test_raw.to_numpy()
    test_r2_scores = {f: r2_score(y_test_final[:, i], y_pred_raw[:, i]) for i, f in enumerate(OUTPUT_FEATURES)}
    
    # Training predictions
    train_preds_scaled = model.predict(X_train_scaled, verbose=0)
    train_pred_raw = scaler_y.inverse_transform(train_preds_scaled)
    y_train_final = y_train_raw.to_numpy()
    train_r2_scores = {f: r2_score(y_train_final[:, i], train_pred_raw[:, i]) for i, f in enumerate(OUTPUT_FEATURES)}
    
    # Full dataset predictions
    full_preds_scaled = model.predict(scaler_X.transform(X), verbose=0)
    full_pred_raw = scaler_y.inverse_transform(full_preds_scaled)
    y_full_final = y_raw.to_numpy()
    full_r2_scores = {f: r2_score(y_full_final[:, i], full_pred_raw[:, i]) for i, f in enumerate(OUTPUT_FEATURES)}
    
    # Prepare dataframes for reports
    train_report_df = pd.DataFrame(y_train_final, index=y_train_raw.index, columns=[f'Actual_{c}' for c in OUTPUT_FEATURES])
    train_report_df[[f'Predicted_{c}' for c in OUTPUT_FEATURES]] = train_pred_raw
    
    test_report_df = pd.DataFrame(y_test_final, index=y_test_raw.index, columns=[f'Actual_{c}' for c in OUTPUT_FEATURES])
    test_report_df[[f'Predicted_{c}' for c in OUTPUT_FEATURES]] = y_pred_raw

    full_report_df = pd.DataFrame(y_full_final, index=y_raw.index, columns=[f'Actual_{c}' for c in OUTPUT_FEATURES])
    full_report_df[[f'Predicted_{c}' for c in OUTPUT_FEATURES]] = full_pred_raw
    
    print(f"  > Test R² for seed {seed} (Entry): {test_r2_scores['Entry dia']:.4f}")
    
    return test_r2_scores, train_r2_scores, full_r2_scores, train_report_df, test_report_df, full_report_df
# --- 2. Data Loading and Preparation (Run Once) ---
try:
    df_main = pd.read_csv('Final.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Final.csv' not found.")
    exit()

drop_cols = ['Sl No. (Image)', 'Unnamed: 6', 'Time (s)', 'OC at Entry', 'OC at Exit']
df_main = df_main.drop(columns=drop_cols, errors='ignore').dropna(subset=INPUT_FEATURES + OUTPUT_FEATURES)
print("Feature engineering complete.")

# --- 3. Execute the 50-Run Experiment ---
print("\n--- Starting 50-Run TensorFlow Evaluation ---")
start_eval_time = time.time()

num_runs = 50
all_test_r2, all_train_r2, all_full_r2 = [{f: [] for f in OUTPUT_FEATURES} for _ in range(3)]
all_train_reports, all_full_reports = [], []
all_test_reports = []
for i in range(num_runs):
    test_scores, train_scores, full_scores, train_report, test_report, full_report = run_tf_experiment(seed=i, df=df_main, best_hps=tuned_params)
    
    
    for feature in OUTPUT_FEATURES:
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
print(f"\n--- Final Aggregated TensorFlow Results from {num_runs} Runs ---")

print("\nMetrics for TEST SET (Primary Paper Result):")
for feature, scores in all_test_r2.items():
    mean_r2, std_r2 = np.mean(scores), np.std(scores)
    print(f"  {feature}: R-squared = {mean_r2:.4f} ± {std_r2:.4f}")

print("\nMetrics for TRAINING SET (For Appendix/Diagnostics):")
for feature, scores in all_train_r2.items():
    print(f"  {feature}: Average R-squared = {np.mean(scores):.4f}")

print("\nMetrics for FULL DATASET (For Appendix/Diagnostics):")
for feature, scores in all_full_r2.items():
    print(f"  {feature}: Average R-squared = {np.mean(scores):.4f}")

# --- 5. Generate and Save CSV Reports ---
print("\n--- Generating CSV Reports ---")

training_report_all_runs = pd.concat(all_train_reports)
training_csv_path = os.path.join(OUTPUT_DIR, 'tf_training_predictions_all_runs.csv')
training_report_all_runs.to_csv(training_csv_path, index=True)
print(f"Training predictions saved to: {training_csv_path}")
# Combine and save TESTING reports
testing_report_all_runs = pd.concat(all_test_reports)
testing_csv_path = os.path.join(OUTPUT_DIR, 'tf_testing_predictions_all_runs.csv')
testing_report_all_runs.to_csv(testing_csv_path, index=True)
print(f"Testing predictions saved to: {testing_csv_path}")

full_report_all_runs = pd.concat(all_full_reports)
full_csv_path = os.path.join(OUTPUT_DIR, 'tf_full_dataset_predictions_all_runs.csv')
full_report_all_runs.to_csv(full_csv_path, index=True)
print(f"Full dataset predictions saved to: {full_csv_path}")

# --- 6. Save Raw R-squared Scores for T-Test Analysis ---
output_dir_2 = 'r2_analysis_results'
os.makedirs(output_dir_2, exist_ok=True)
raw_scores_filename = os.path.join(output_dir_2, 'raw_r2_scores_tensorflow.json')
with open(raw_scores_filename, 'w') as f:
    json.dump(all_test_r2, f, indent=4)

print(f"Raw R2 scores saved to: {raw_scores_filename}")

# --- 7. Per-Seed R² with Confidence Intervals ---
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

# 1) TEST 20% — per-seed R², then mean/std/CI
df_test = pd.read_csv(testing_csv_path, index_col=0)
r2_entry_test = per_seed_r2(df_test, 'Actual_Entry dia', 'Predicted_Entry dia')
r2_exit_test  = per_seed_r2(df_test, 'Actual_Exit dia',  'Predicted_Exit dia')

mean_e_t, std_e_t, ci_e_t = summarize(r2_entry_test['r2'])
mean_x_t, std_x_t, ci_x_t = summarize(r2_exit_test['r2'])

print("TensorFlow — TEST 20% R² across seeds (mean ± std, 95% CI)")
print(f"Entry dia: mean={mean_e_t:.4f}, std={std_e_t:.4f}, 95% CI=({ci_e_t[0]:.4f}, {ci_e_t[1]:.4f})")
print(f"Exit dia : mean={mean_x_t:.4f}, std={std_x_t:.4f}, 95% CI=({ci_x_t[0]:.4f}, {ci_x_t[1]:.4f})")

r2_entry_test.to_csv('r2_analysis_results/tf_test_r2_per_seed_entry.csv', index=False)
r2_exit_test.to_csv('r2_analysis_results/tf_test_r2_per_seed_exit.csv', index=False)

# 2) FULL 112 — per-seed R², then mean/std/CI
df_full = pd.read_csv(full_csv_path, index_col=0)

rows_entry, rows_exit = [], []
for seed, g in df_full.groupby('run'):
    r2e = r2_score(g['Actual_Entry dia'], g['Predicted_Entry dia'])
    r2x = r2_score(g['Actual_Exit dia'],  g['Predicted_Exit dia'])
    rows_entry.append((seed, r2e))
    rows_exit.append((seed, r2x))

r2_entry_full_seeds = pd.DataFrame(rows_entry, columns=['run','r2']).sort_values('run', ignore_index=True)
r2_exit_full_seeds  = pd.DataFrame(rows_exit,  columns=['run','r2']).sort_values('run', ignore_index=True)

mean_e_f, std_e_f, ci_e_f = summarize(r2_entry_full_seeds['r2'])
mean_x_f, std_x_f, ci_x_f = summarize(r2_exit_full_seeds['r2'])

print("\nTensorFlow — FULL 112 R² across seeds (mean ± std, 95% CI)")
print(f"Entry dia: mean={mean_e_f:.4f}, std={std_e_f:.4f}, 95% CI=({ci_e_f[0]:.4f}, {ci_e_f[1]:.4f})")
print(f"Exit dia : mean={mean_x_f:.4f}, std={std_x_f:.4f}, 95% CI=({ci_x_f[0]:.4f}, {ci_x_f[1]:.4f})")

r2_entry_full_seeds.to_csv('r2_analysis_results/tf_full_r2_per_seed_entry.csv', index=False)
r2_exit_full_seeds.to_csv('r2_analysis_results/tf_full_r2_per_seed_exit.csv', index=False)