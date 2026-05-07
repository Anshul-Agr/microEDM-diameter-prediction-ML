import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import os
import json
from tqdm import tqdm
from scipy import stats as scipy_stats

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'font.size': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

# --- 0. Data Loading ---
df = pd.read_csv('Final.csv')

input_features = ['Cap (pF)', 'Vol (V)', 'TRS (rpm)', 'FR (µm/s)']
output_features = ['Entry dia', 'Exit dia']
columns_original = ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)', 'Entry dia', 'Exit dia']
columns_new = ['Cap (pF)', 'Vol (V)', 'TRS (rpm)', 'FR (µm/s)', 'Entry dia', 'Exit dia']
df.columns = [col.strip() for col in df.columns]
df = df.rename(columns=dict(zip(columns_original, columns_new)))

columns_to_drop = ['Sl No. (Image)', 'Unnamed: 6', 'OC at Entry', 'OC at Exit',
                   'MRR', 'Taper(degree)', 'Time (s)', 'Exp No.']
df = df.drop(columns=columns_to_drop, errors='ignore')

X = df[input_features]
y = df[output_features]

try:
    with open('hyperparameter_tuning_results/xgboost_best_params.json', 'r') as f:
        best_params = json.load(f)
    print("Loaded best hyperparameters.")
except FileNotFoundError:
    print("Warning: Could not load best params, using defaults.")
    best_params = {
        'Entry dia': {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 5},
        'Exit dia': {'n_estimators': 1000, 'learning_rate': 0.03, 'max_depth': 5}
    }

os.makedirs('feature_importance_results', exist_ok=True)

# --- 1. Permutation Importance Across 50 Seeds ---
print("Calculating permutation importance across 50 seeds...")

all_results = {target: {feature: [] for feature in input_features} for target in output_features}
seed_summary = []

for seed in tqdm(range(50), desc="Processing seeds"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    seed_importances = {'seed': seed}

    for target_idx, target in enumerate(output_features):
        params = best_params[target].copy()
        params['random_state'] = seed
        params['n_jobs'] = 1

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train_scaled[:, target_idx], verbose=False)

        perm_result = permutation_importance(
            model, X_test_scaled, y_test_scaled[:, target_idx],
            n_repeats=30, random_state=42, scoring='r2', n_jobs=-1
        )

        for i, feature in enumerate(input_features):
            importance = perm_result.importances_mean[i]
            all_results[target][feature].append(importance)
            seed_importances[f'{target}_{feature}'] = importance

    seed_summary.append(seed_importances)

print("Completed all 50 seeds.")

# --- 2. Aggregated Results ---
for target in output_features:
    print(f"\nTarget: {target}")
    print("-" * 70)

    importance_stats = []
    for feature in input_features:
        importances = np.array(all_results[target][feature])
        mean_imp = np.mean(importances)
        std_imp = np.std(importances)
        median_imp = np.median(importances)
        ci_95 = scipy_stats.t.interval(0.95, len(importances)-1,
                                       loc=mean_imp,
                                       scale=scipy_stats.sem(importances))
        t_stat, p_value = scipy_stats.ttest_1samp(importances, 0)

        importance_stats.append({
            'feature': feature, 'mean': mean_imp, 'std': std_imp,
            'median': median_imp, 'ci_lower': ci_95[0], 'ci_upper': ci_95[1],
            't_stat': t_stat, 'p_value': p_value
        })

    importance_stats.sort(key=lambda x: x['mean'], reverse=True)

    for stat in importance_stats:
        sig = "***" if stat['p_value'] < 0.001 else "**" if stat['p_value'] < 0.01 else "*" if stat['p_value'] < 0.05 else "ns"
        print(f"  {stat['feature']:25s}: {stat['mean']:7.4f} +/- {stat['std']:.4f}  "
              f"[95% CI: {stat['ci_lower']:7.4f}, {stat['ci_upper']:7.4f}]  "
              f"p={stat['p_value']:.6f} {sig:3s}")

# --- 3. Statistical Interpretation ---
for target in output_features:
    print(f"\n{target}:")
    significant_features = []
    negligible_features = []

    for feature in input_features:
        importances = np.array(all_results[target][feature])
        mean_imp = np.mean(importances)
        _, p_value = scipy_stats.ttest_1samp(importances, 0)

        if p_value < 0.05 and mean_imp > 0.01:
            significant_features.append((feature, mean_imp))
        elif p_value >= 0.05:
            negligible_features.append(feature)

    if significant_features:
        print(f"  Significant features (p < 0.05):")
        for feat, imp in sorted(significant_features, key=lambda x: x[1], reverse=True):
            print(f"    {feat}: {imp:.4f}")

    if negligible_features:
        print(f"  Negligible features (p >= 0.05):")
        for feat in negligible_features:
            print(f"    {feat}")

# --- 4. Visualizations ---
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, target in enumerate(output_features):
    # Box plot
    ax_left = axes[idx, 0]
    data_for_box = [all_results[target][feat] for feat in input_features]
    bp = ax_left.boxplot(data_for_box, tick_labels=input_features,
                          patch_artist=True, showmeans=True)
    for i, patch in enumerate(bp['boxes']):
        mean_val = np.mean(data_for_box[i])
        if mean_val < -0.01:
            patch.set_facecolor('lightcoral')
        elif mean_val > 0.01:
            patch.set_facecolor('lightblue')
        else:
            patch.set_facecolor('lightgray')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    ax_left.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_left.set_ylabel('Permutation Importance (R2 drop)', fontweight='bold')
    ax_left.set_title(f'{target}: Distribution Across 50 Seeds', fontweight='bold')
    ax_left.grid(True, alpha=0.3, axis='y')
    ax_left.tick_params(axis='x', rotation=45)

    # Bar chart with CI
    ax_right = axes[idx, 1]
    means = [np.mean(all_results[target][feat]) for feat in input_features]
    cis = []
    for feat in input_features:
        data = all_results[target][feat]
        ci = scipy_stats.t.interval(0.95, len(data)-1, loc=np.mean(data),
                                    scale=scipy_stats.sem(data))
        cis.append((ci[1] - np.mean(data)))

    sorted_indices = np.argsort(means)
    y_pos = np.arange(len(input_features))
    colors = ['red' if m < -0.01 else 'green' if m > 0.01 else 'gray'
              for m in np.array(means)[sorted_indices]]

    ax_right.barh(y_pos, np.array(means)[sorted_indices],
                  xerr=np.array(cis)[sorted_indices],
                  alpha=0.7, color=colors, edgecolor='black', linewidth=1.5,
                  error_kw={'linewidth': 2, 'elinewidth': 2})

    ax_right.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([input_features[i] for i in sorted_indices], fontweight='bold')
    ax_right.set_xlabel('Mean Importance +/- 95% CI', fontweight='bold')
    ax_right.set_title(f'{target}: Mean Across 50 Seeds', fontweight='bold')
    ax_right.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance_results/permutation_50seeds_rigorous.png',
            dpi=600, bbox_inches='tight', pad_inches=0.1)
print("Saved: permutation_50seeds_rigorous.png")
plt.close()

# Heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for idx, target in enumerate(output_features):
    ax = axes[idx]
    data_matrix = np.array([all_results[target][feat] for feat in input_features]).T
    sns.heatmap(data_matrix,
                xticklabels=input_features,
                yticklabels=[f'Seed {i}' if i % 10 == 0 else '' for i in range(50)],
                cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={'label': 'Importance'}, ax=ax)
    ax.set_title(f'{target}: Importance Across All Seeds', fontweight='bold')
    ax.set_xlabel('Features', fontweight='bold')
    ax.set_ylabel('Seeds', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_results/permutation_heatmap_seeds.png',
            dpi=600, bbox_inches='tight', pad_inches=0.1)
print("Saved: permutation_heatmap_seeds.png")
plt.close()

# --- 5. Boxplot Statistics Export ---
box_stats_rows = []
for target in output_features:
    for feature in input_features:
        data = np.array(all_results[target][feature], dtype=float)
        if data.size == 0:
            continue
        q1 = float(np.percentile(data, 25))
        median = float(np.percentile(data, 50))
        q3 = float(np.percentile(data, 75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        whisker_low = float(np.min(data[data >= lower_fence])) if np.any(data >= lower_fence) else float(np.min(data))
        whisker_high = float(np.max(data[data <= upper_fence])) if np.any(data <= upper_fence) else float(np.max(data))

        box_stats_rows.append({
            'Target': target, 'Feature': feature, 'N_Seeds': int(data.size),
            'Raw_Min': float(np.min(data)), 'Q1_25th': q1, 'Median_50th': median,
            'Q3_75th': q3, 'Raw_Max': float(np.max(data)), 'IQR': iqr,
            'Lower_Fence': lower_fence, 'Upper_Fence': upper_fence,
            'Whisker_Low': whisker_low, 'Whisker_High': whisker_high
        })

box_stats_df = pd.DataFrame(box_stats_rows)
box_stats_df.to_csv('feature_importance_results/permutation_boxplot_stats_50seeds.csv', index=False)
print("Saved: permutation_boxplot_stats_50seeds.csv")

# --- 6. Export Results ---
seed_df = pd.DataFrame(seed_summary)
seed_df.to_csv('feature_importance_results/permutation_per_seed_50seeds.csv', index=False)

summary_data = []
for target in output_features:
    for feature in input_features:
        importances = np.array(all_results[target][feature])
        t_stat, p_value = scipy_stats.ttest_1samp(importances, 0)
        ci = scipy_stats.t.interval(0.95, len(importances)-1,
                                    loc=np.mean(importances),
                                    scale=scipy_stats.sem(importances))
        summary_data.append({
            'Target': target, 'Feature': feature,
            'Mean_Importance': np.mean(importances),
            'Std_Importance': np.std(importances),
            'Median_Importance': np.median(importances),
            'CI_95_Lower': ci[0], 'CI_95_Upper': ci[1],
            'T_Statistic': t_stat, 'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('feature_importance_results/permutation_summary_50seeds.csv', index=False)
print("Saved: permutation_summary_50seeds.csv")
print("\nPermutation importance analysis complete.")
