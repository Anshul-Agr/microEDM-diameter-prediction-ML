import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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
y_train_scaled_df = pd.DataFrame(y_train_scaled, index=y_train.index, columns=output_features)

# --- 3. Hyperparameter Search (per-target) ---
param_grid = {
    'n_estimators': [500, 1000, 2000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.01, 0.1]
}

final_best_params = {}
print("\n--- Starting Hyperparameter Tuning for XGBoost ---")
start_tuning_time = time.time()

for feature in output_features:
    print(f"\n>>> Tuning for target: {feature}...")

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=TUNING_SEED)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=TUNING_SEED,
        verbose=1
    )

    random_search.fit(X_train_scaled, y_train_scaled_df[feature])
    final_best_params[feature] = random_search.best_params_

    print(f"  Best R2 for {feature}: {random_search.best_score_:.4f}")
    print(f"  Best parameters: {random_search.best_params_}")

total_tuning_time = time.time() - start_tuning_time

# --- 4. Save Results ---
output_path = os.path.join(OUTPUT_DIR, 'xgboost_best_params.json')
with open(output_path, 'w') as f:
    json.dump(final_best_params, f, indent=4)

print(f"\n--- Hyperparameter Tuning Complete ---")
print(f"  Time: {total_tuning_time:.2f} seconds")
print(f"  Saved to: {output_path}")
