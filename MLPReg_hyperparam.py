import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import os
import json
import time

# --- 0. Global Setup ---
TUNING_SEED = 42
OUTPUT_DIR = 'hyperparameter_tuning_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

input_features = ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)']
output_features = ['Entry dia', 'Exit dia', 'MRR', 'Taper(degree)']

# --- 1. Data Loading and Preparation ---
try:
    df_main = pd.read_csv('Final.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Final.csv' not found.")
    exit()

columns_to_drop = ['Sl No. (Image)', 'Unnamed: 6', 'OC at Entry', 'OC at Exit']
df_main = df_main.drop(columns=columns_to_drop, errors='ignore')

# --- 2. Data Splits and Scaling ---
X = df_main[input_features]
y = df_main[output_features]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=TUNING_SEED)

scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
scaler_y = StandardScaler().fit(y_train)
y_train_scaled = scaler_y.transform(y_train)

# --- 3. Hyperparameter Search ---
param_distributions = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'batch_size': [8, 16, 32, 64, 'auto']
}

print("\n--- Starting Hyperparameter Tuning for MLP ---")
start_tuning_time = time.time()

mlp = MLPRegressor(
    random_state=TUNING_SEED,
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    verbose=False
)

random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=30,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=TUNING_SEED
)

random_search.fit(X_train_scaled, y_train_scaled)
total_tuning_time = time.time() - start_tuning_time

# --- 4. Save Results ---
best_params = random_search.best_params_
output_path = os.path.join(OUTPUT_DIR, 'mlp_best_params.json')
with open(output_path, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"\n--- Hyperparameter Tuning Complete ---")
print(f"  Time: {total_tuning_time:.2f} seconds")
print(f"  Best R2: {random_search.best_score_:.4f}")
print(f"  Best parameters: {best_params}")
print(f"  Saved to: {output_path}")
