import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import time

# --- 0. Global Setup ---
TUNING_SEED = 42
tf.random.set_seed(TUNING_SEED)
np.random.seed(TUNING_SEED)
OUTPUT_DIR = 'hyperparameter_tuning_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FEATURES = ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)']
OUTPUT_FEATURES = ['Entry dia', 'Exit dia', 'MRR', 'Taper(degree)']

# --- 1. Data Loading and Preparation ---
try:
    df_main = pd.read_csv('Final.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Final.csv' not found.")
    exit()

columns_to_drop = ['Sl No. (Image)', 'Unnamed: 6', 'OC at Entry', 'OC at Exit']
df_main = df_main.drop(columns=columns_to_drop, errors='ignore')
df_main = df_main.dropna(subset=INPUT_FEATURES + OUTPUT_FEATURES)

# --- 2. Data Splits and Scaling ---
X = df_main[INPUT_FEATURES]
y = df_main[OUTPUT_FEATURES]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=TUNING_SEED)

scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

scaler_y = StandardScaler().fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)

# --- 3. Model Search Space ---
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))

    model.add(tf.keras.layers.Dense(len(OUTPUT_FEATURES), activation='linear'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])
    auto_batch_value = X_train_scaled.shape[0]
    hp.Choice('batch_size', values=[8, 16, 32, 64, auto_batch_value])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error'
    )
    return model

# --- 4. Hyperparameter Search ---
class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.get('batch_size')
        return super(MyTuner, self).run_trial(trial, *args, **kwargs)

tuner = MyTuner(
    build_model,
    objective='val_loss',
    max_epochs=200,
    factor=3,
    directory=os.path.join(OUTPUT_DIR, 'keras_tuner_dir'),
    project_name='tf_tuning',
    overwrite=True
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)

print("\n--- Starting Hyperparameter Tuning for TensorFlow ---")
start_tuning_time = time.time()

tuner.search(
    X_train_scaled,
    y_train_scaled,
    epochs=200,
    validation_data=(X_val_scaled, y_val_scaled),
    callbacks=[stop_early]
)

total_tuning_time = time.time() - start_tuning_time

# --- 5. Save Best Hyperparameters ---
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\n--- Hyperparameter Tuning Complete ---")
print(f"  Time: {total_tuning_time:.2f} seconds")
print(f"  Layers: {best_hps.get('num_layers')}")
print(f"  Activation: {best_hps.get('activation')}")
print(f"  Learning rate: {best_hps.get('learning_rate')}")
for i in range(best_hps.get('num_layers')):
    print(f"  Units in layer {i}: {best_hps.get(f'units_{i}')}")

params_to_save = {k: v for k, v in best_hps.values.items()}
output_path = os.path.join(OUTPUT_DIR, 'tensorflow_best_params.json')
with open(output_path, 'w') as f:
    json.dump(params_to_save, f, indent=4)
print(f"  Saved to: {output_path}")

# --- 6. Find Optimal Epoch Count ---
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

print("\nRe-training best model to find optimal epoch count...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    validation_data=(X_val_scaled, y_val_scaled),
    batch_size=best_hps.get('batch_size'),
    callbacks=[stop_early],
    verbose=0
)

val_loss_history = history.history['val_loss']
best_epoch = val_loss_history.index(min(val_loss_history)) + 1

print(f"  Optimal batch size: {best_hps.get('batch_size')}")
print(f"  Optimal epoch count: {best_epoch}")

final_report = {
    "best_batch_size": int(best_hps.get('batch_size')),
    "optimal_epochs": best_epoch,
    "final_val_loss": float(min(val_loss_history))
}

with open(os.path.join(OUTPUT_DIR, 'final_convergence_report.json'), 'w') as f:
    json.dump(final_report, f, indent=4)
print(f"  Convergence report: {final_report}")