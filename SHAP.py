import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import json
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

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
    print("Error loading best params.")
    exit()

os.makedirs('shap_clean/main_figures', exist_ok=True)
os.makedirs('shap_clean/supplementary', exist_ok=True)
os.makedirs('shap_clean/data', exist_ok=True)

# --- 1. Train Models and Calculate SHAP Values (50 seeds) ---
print("Training models and calculating SHAP values across 50 seeds...")

N_SEEDS = 50
shap_all_seeds = {target: [] for target in output_features}
test_data_all = {target: [] for target in output_features}

for seed in tqdm(range(N_SEEDS), desc="Seeds"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    for target_idx, target in enumerate(output_features):
        params = best_params[target].copy()
        params['random_state'] = seed
        params['n_jobs'] = 1

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train_scaled[:, target_idx], verbose=False)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        shap_all_seeds[target].append(shap_values)
        test_data_all[target].append({
            'X_test': X_test,
            'X_test_scaled': X_test_scaled,
            'base_value': explainer.expected_value,
            'model': model
        })

print("All models trained and SHAP values calculated.")

# --- 2. Aggregate SHAP Statistics ---
print("Aggregating SHAP statistics...")

aggregated_shap = {}

for target in output_features:
    seed_importance_list = []
    for s in range(N_SEEDS):
        importances = np.abs(shap_all_seeds[target][s]).mean(axis=0)
        seed_importance_list.append(importances)

    seed_importances = np.array(seed_importance_list)

    median_abs = np.median(seed_importances, axis=0)
    std_abs = np.std(seed_importances, axis=0)
    q1 = np.percentile(seed_importances, 25, axis=0)
    q3 = np.percentile(seed_importances, 75, axis=0)
    iqr = q3 - q1
    ci_lower = np.percentile(seed_importances, 2.5, axis=0)
    ci_upper = np.percentile(seed_importances, 97.5, axis=0)

    all_shap = np.concatenate(shap_all_seeds[target], axis=0)
    all_X_test = pd.concat([test_data_all[target][i]['X_test'] for i in range(N_SEEDS)], axis=0)

    aggregated_shap[target] = {
        'shap_values': all_shap,
        'X_test': all_X_test,
        'seed_importances': seed_importances,
        'median': median_abs,
        'std_abs': std_abs,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_abs': np.mean(seed_importances, axis=0),
        'base_value': np.mean([test_data_all[target][i]['base_value'] for i in range(N_SEEDS)])
    }

    print(f"\nTarget: {target}")
    print(f"{'Feature':<15} | {'Median':<10} | {'95% CI Lower':<12} | {'95% CI Upper':<12} | {'IQR':<8}")
    print("-" * 70)
    for i, feat in enumerate(input_features):
        print(f"{feat:<15} | {median_abs[i]:.4f} | {ci_lower[i]:.4f}   | {ci_upper[i]:.4f}   | {iqr[i]:.4f}")

# --- 3. Figure 1: Feature Importance Bar Plot ---
print("\nGenerating Figure 1: Feature importance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, target in enumerate(output_features):
    ax = axes[idx]
    mean_abs = aggregated_shap[target]['mean_abs']
    std_abs = aggregated_shap[target]['std_abs']
    sorted_idx = np.argsort(mean_abs)

    y_pos = np.arange(len(input_features))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(input_features)))
    err_low = mean_abs - aggregated_shap[target]['ci_lower']
    err_high = aggregated_shap[target]['ci_upper'] - mean_abs
    asymmetric_error = [err_low[sorted_idx], err_high[sorted_idx]]

    bars = ax.barh(y_pos, mean_abs[sorted_idx],
                   xerr=asymmetric_error,
                   color=[colors[i] for i in sorted_idx],
                   edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 2, 'capsize': 5, 'capthick': 2})

    ax.set_yticks(y_pos)
    ax.set_yticklabels([input_features[i] for i in sorted_idx], fontweight='bold')
    ax.set_xlabel('Mean |SHAP value| & 95% CI', fontweight='bold')
    ax.set_title(f'{target}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig1_SHAP_importance.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig('shap_clean/main_figures/Fig1_SHAP_importance.pdf', bbox_inches='tight')
print("Saved: Fig1_SHAP_importance")
plt.close()

# --- 4. Figure 2: Beeswarm Scatter ---
print("Generating Figure 2: Beeswarm scatter...")

fig, axes = plt.subplots(2, 1, figsize=(12, 12))

for idx, target in enumerate(output_features):
    ax = axes[idx]
    shap_vals = aggregated_shap[target]['shap_values']
    X_test = aggregated_shap[target]['X_test']

    y_positions = []
    colors_list = []
    shap_list = []

    for feat_idx, feature in enumerate(input_features):
        feat_shap = shap_vals[:, feat_idx]
        feat_values = X_test[feature].values
        feat_norm = (feat_values - feat_values.min()) / (feat_values.max() - feat_values.min() + 1e-10)
        y_positions.extend([feat_idx] * len(feat_shap))
        colors_list.extend(feat_norm)
        shap_list.extend(feat_shap)

    scatter = ax.scatter(shap_list, y_positions,
                        c=colors_list, cmap='coolwarm',
                        alpha=0.6, s=20, edgecolors='black', linewidths=0.3)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_yticks(range(len(input_features)))
    ax.set_yticklabels(input_features, fontweight='bold')
    ax.set_xlabel('SHAP value (impact on prediction)', fontweight='bold')
    ax.set_title(f'{target}: Feature Effects', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Feature Value\n(Low -> High)', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig2_SHAP_beeswarm.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig('shap_clean/main_figures/Fig2_SHAP_beeswarm.pdf', bbox_inches='tight')
print("Saved: Fig2_SHAP_beeswarm")
plt.close()

# --- 5. Figure 3: Dependence Plots (Top 2 Features) ---
print("Generating Figure 3: Dependence plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for target_idx, target in enumerate(output_features):
    shap_vals = aggregated_shap[target]['shap_values']
    X_test = aggregated_shap[target]['X_test']

    mean_abs = aggregated_shap[target]['mean_abs']
    top_2_idx = np.argsort(mean_abs)[-2:][::-1]
    top_2_features = [input_features[i] for i in top_2_idx]

    for feat_idx, (feat_pos, feature) in enumerate(zip(top_2_idx, top_2_features)):
        if target_idx == 0:
            col = feat_idx
        else:
            col = 1 - feat_idx

        ax = axes[target_idx, col]

        feat_shap = shap_vals[:, feat_pos]
        feat_values = X_test[feature].values

        correlations = []
        for other_idx, other_feat in enumerate(input_features):
            if other_idx != feat_pos:
                corr = np.corrcoef(X_test[other_feat].values, feat_shap)[0, 1]
                correlations.append((abs(corr), other_idx, other_feat))

        _, interact_idx, interact_feat = max(correlations)
        interact_values = X_test[interact_feat].values

        scatter = ax.scatter(
            feat_values, feat_shap,
            c=interact_values,
            cmap='viridis',
            vmin=interact_values.min(),
            vmax=interact_values.max(),
            alpha=0.6, s=30, edgecolors='black', linewidths=0.5
        )

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel(feature, fontweight='bold')
        ax.set_ylabel(f'SHAP value for {target}', fontweight='bold')
        ax.set_title(f'{target}: {feature} Effect', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(f'{interact_feat}\n(interaction)', fontweight='bold')

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig3_SHAP_dependence.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig('shap_clean/main_figures/Fig3_SHAP_dependence.pdf', bbox_inches='tight')
print("Saved: Fig3_SHAP_dependence")
plt.close()

# --- 6. Figure 4: Waterfall Plot (Global Worst Outlier) ---
print("Generating Figure 4: Waterfall (worst outlier)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for target_idx, target in enumerate(output_features):
    worst_error = -np.inf
    for seed in range(N_SEEDS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        params = best_params[target].copy()
        params['random_state'] = seed
        params['n_jobs'] = 1
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train_scaled[:, target_idx], verbose=False)
        predictions_scaled = model.predict(X_test_scaled)
        predictions = predictions_scaled * scaler_y.scale_[target_idx] + scaler_y.mean_[target_idx]
        actual = y_test[target].values
        residuals = actual - predictions
        max_idx = np.argmax(np.abs(residuals))
        max_error = np.abs(residuals[max_idx])
        if max_error > worst_error:
            worst_error = max_error
            best_seed = seed
            best_model = model
            best_X_test = X_test
            best_X_test_scaled = X_test_scaled
            best_scaler_y = scaler_y
            best_target_idx = target_idx
            best_actual = actual
            best_predictions = predictions
            best_worst_idx = max_idx
            best_y_test = y_test

    ax = axes[target_idx]
    explainer = shap.TreeExplainer(best_model)
    shap_worst = explainer.shap_values(best_X_test_scaled[best_worst_idx:best_worst_idx+1])[0]
    base_value = explainer.expected_value
    shap_worst = shap_worst * best_scaler_y.scale_[best_target_idx]
    base_value = base_value * best_scaler_y.scale_[best_target_idx] + best_scaler_y.mean_[best_target_idx]
    feature_values = best_X_test.iloc[best_worst_idx][input_features].values
    sorted_indices = np.argsort(np.abs(shap_worst))[::-1]
    cumsum = np.cumsum([base_value] + list(shap_worst[sorted_indices]))
    y_pos = np.arange(len(input_features) + 2)
    ax.barh(0, base_value, color='gray', alpha=0.3, edgecolor='black', linewidth=1.5)
    ax.text(base_value/2, 0, f'Base\n{base_value:.1f}', ha='center', va='center', fontweight='bold')
    for i, idx in enumerate(sorted_indices):
        feature = input_features[idx]
        shap_val = shap_worst[idx]
        feat_val = feature_values[idx]
        color = 'salmon' if shap_val < 0 else 'lightblue'
        left = cumsum[i]
        ax.barh(i+1, shap_val, left=left, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.plot([cumsum[i+1], cumsum[i+1]], [i, i+1], 'k--', linewidth=0.8, alpha=0.3)
        ax.text(cumsum[i+1], i+1, f'  {shap_val:+.1f}', va='center', fontweight='bold')
        ax.text(left-5, i+1, f'{feature}\n{feat_val:.0f}', va='center', ha='right')
    final_pred = cumsum[-1]
    ax.barh(len(input_features)+1, final_pred, color='gold', alpha=0.3, edgecolor='black', linewidth=1.5)
    ax.text(final_pred/2, len(input_features)+1, f'Predicted\n{final_pred:.1f}', ha='center', va='center', fontweight='bold', fontsize=9)
    ax.axvline(best_actual[best_worst_idx], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Actual: {best_actual[best_worst_idx]:.1f}')
    ax.set_yticks(y_pos)
    labels = ['Base'] + [f'{i+1}' for i in range(len(input_features))] + ['Final']
    ax.set_yticklabels(labels)
    ax.set_xlabel('Diameter (um)', fontweight='bold')
    ax.set_title(f'{target}: Worst Case Explanation\nSeed={best_seed}, Error={worst_error:.2f} um', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig4_SHAP_waterfall_outlier.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.savefig('shap_clean/main_figures/Fig4_SHAP_waterfall_outlier.pdf', bbox_inches='tight')
print("Saved: Fig4_SHAP_waterfall_outlier")
plt.close()

# --- 7. Supplementary: All Feature Dependence Plots ---
print("Generating supplementary dependence plots...")

for target in output_features:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    shap_vals = aggregated_shap[target]['shap_values']
    X_test = aggregated_shap[target]['X_test']

    for feat_idx, feature in enumerate(input_features):
        ax = axes[feat_idx]
        feat_shap = shap_vals[:, feat_idx]
        feat_values = X_test[feature].values
        ax.scatter(feat_values, feat_shap, alpha=0.5, s=20,
                  edgecolors='black', linewidths=0.3, c='steelblue')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel(feature, fontweight='bold', fontsize=13)
        ax.set_ylabel('SHAP value', fontweight='bold', fontsize=13)
        ax.set_title(f'{feature} Effect', fontweight='bold', fontsize=13)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'{target}: Complete Dependence Analysis', fontweight='bold', fontsize=14)
    plt.tight_layout()
    filename = f"SuppFig_all_dependence_{target.replace(' ', '_')}.png"
    plt.savefig(f'shap_clean/supplementary/{filename}', dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {filename}")
    plt.close()

# --- 8. Export SHAP Data ---
print("Exporting SHAP data...")

summary_data = []
for target in output_features:
    mean_abs = aggregated_shap[target]['mean_abs']
    std_abs = aggregated_shap[target]['std_abs']
    for i, feature in enumerate(input_features):
        summary_data.append({
            'Target': target,
            'Feature': feature,
            'Mean_Abs_SHAP': mean_abs[i],
            'Std_Abs_SHAP': std_abs[i],
            'Percentage': 100 * mean_abs[i] / mean_abs.sum()
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('shap_clean/data/SHAP_summary_50seeds.csv', index=False)

for target in output_features:
    shap_df = pd.DataFrame(aggregated_shap[target]['shap_values'], columns=input_features)
    filename = f"SHAP_raw_values_{target.replace(' ', '_')}.csv"
    shap_df.to_csv(f'shap_clean/data/{filename}', index=False)

print("Saved: SHAP summary and raw values")

# --- 9. Figure 5: Importance Stability Boxplot ---
print("Generating Figure 5: Importance stability boxplot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, target in enumerate(output_features):
    ax = axes[idx]
    data = aggregated_shap[target]['seed_importances']
    medians = aggregated_shap[target]['median']
    sorted_idx = np.argsort(medians)
    sorted_data = data[:, sorted_idx]
    sorted_labels = [input_features[i] for i in sorted_idx]

    bp = ax.boxplot(sorted_data, vert=False, patch_artist=True,
                    labels=sorted_labels, showmeans=True,
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(input_features)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(f'{target}: Importance Stability\n(N=50 Seeds)', fontweight='bold')
    ax.set_xlabel('Mean |SHAP value|', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig5_Importance_Stability_Boxplot.png', dpi=600)
plt.savefig('shap_clean/main_figures/Fig5_Importance_Stability_Boxplot.pdf')
print("Saved: Fig5_Importance_Stability_Boxplot")
plt.close()

# --- 10. Figure 6: Trend Stability ---
print("Generating Figure 6: Trend stability...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for target_idx, target in enumerate(output_features):
    shap_vals = aggregated_shap[target]['shap_values']
    X_test = aggregated_shap[target]['X_test']

    mean_abs = aggregated_shap[target]['mean_abs']
    top_2_idx = np.argsort(mean_abs)[-2:][::-1]

    for feat_idx, feat_pos in enumerate(top_2_idx):
        ax = axes[target_idx, feat_idx]
        feature = input_features[feat_pos]

        sns.regplot(x=X_test[feature].values, y=shap_vals[:, feat_pos],
                    ax=ax, scatter_kws={'alpha':0.15, 's':15, 'color':'#2c3e50'},
                    line_kws={'color':'#e74c3c', 'lw':3}, lowess=True)

        ax.axhline(0, color='black', linestyle='-', alpha=0.2)
        ax.set_xlabel(f'{feature} (Input Value)', fontweight='bold')
        ax.set_ylabel(f'SHAP Value (Impact on {target})', fontweight='bold')
        ax.set_title(f'{target}: {feature} Trend Stability', fontweight='bold')
        ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('shap_clean/main_figures/Fig6_Trend_Stability.png', dpi=600)
plt.savefig('shap_clean/main_figures/Fig6_Trend_Stability.pdf')
print("Saved: Fig6_Trend_Stability")
plt.close()

# --- 11. Robustness Analysis Table ---
print("Exporting robustness analysis table...")
robust_rows = []

for target in output_features:
    stats = aggregated_shap[target]
    for i, feature in enumerate(input_features):
        stability = stats['iqr'][i] / (stats['median'][i] + 1e-10)
        robust_rows.append({
            'Target Variable': target,
            'Input Feature': feature,
            'Median (|SHAP|)': f"{stats['median'][i]:.4f}",
            'IQR': f"{stats['iqr'][i]:.4f}",
            '95% CI Lower': f"{stats['ci_lower'][i]:.4f}",
            '95% CI Upper': f"{stats['ci_upper'][i]:.4f}",
            'Stability Index (IQR/Med)': f"{stability:.3f}"
        })

robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv('shap_clean/data/Table_Robustness_Analysis_50seeds.csv', index=False)
print("Saved: Table_Robustness_Analysis_50seeds.csv")
print("\nSHAP analysis complete.")