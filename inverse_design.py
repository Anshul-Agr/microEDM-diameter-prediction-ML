import os
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
PLOT_DIR = "INVERSE_plots"
os.makedirs(PLOT_DIR, exist_ok=True)
RESULTS_DIR = "INVERSE_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 0. Paths and global config
# ----------------------------------------------------------------------

DATA_PATH = "Final.csv"
SURROGATE_PATH = os.path.join("xgboost_multi_output_model",
                              "xgboost_surrogate_seed0.joblib")

# ----------------------------------------------------------------------
# 1. Load data and surrogate bundle
# ----------------------------------------------------------------------


df = pd.read_csv(DATA_PATH)


columns_to_drop = [
    'Sl No. (Image)',
    'Unnamed: 6',
    'OC at Entry',
    'OC at Exit',
    'MRR',
    'Taper(degree)', 
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# (Optional) print Den/Dex range to sanity-check targets
print(f"Den range: {df['Entry dia'].min():.2f}–{df['Entry dia'].max():.2f} µm")
print(f"Dex range: {df['Exit dia'].min():.2f}–{df['Exit dia'].max():.2f} µm")

# Load surrogate bundle (models + scalers)
bundle = joblib.load(SURROGATE_PATH)
models = bundle["models"]          # dict: target -> XGBRegressor
scaler_X = bundle["scaler_X"]
scaler_y = bundle["scaler_y"]
input_features = bundle["input_features"]
output_features = bundle["output_features"]


assert 'Entry dia' in output_features and 'Exit dia' in output_features, \
    "Surrogate bundle must contain Entry dia and Exit dia models."


den_std = float(df['Entry dia'].std(ddof=1))
dex_std = float(df['Exit dia'].std(ddof=1))


# ----------------------------------------------------------------------
# 2. Helper: discrete design space and taper geometry
# ----------------------------------------------------------------------

def get_default_levels(df: pd.DataFrame):
    """Sorted unique levels for each controllable variable."""
    levels = {}
    for col in ['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)', 'Feed rate (µm/s)']:
        levels[col] = sorted(df[col].unique())
    return levels

DEFAULT_LEVELS = get_default_levels(df)


HOLE_DEPTH = 700.0  # µm

def compute_taper_angle(entry_d: float, exit_d: float, depth: float = HOLE_DEPTH) -> float:
    """
    Compute full included taper angle (theta) in radians
    from entry and exit diameters and hole depth.

    theta = atan((D_entry - D_exit) / (2 * depth))
    """
    return np.arctan((entry_d - exit_d) / (2.0 * depth))


# ----------------------------------------------------------------------
# 3. Core evaluator: predict Entry, Exit, Taper for given settings
# ----------------------------------------------------------------------

def predict_geometry(cap: float, vol: float, trs: float, feed: float) -> tuple[float, float, float]:
    """
    Single-point prediction of Entry, Exit and Taper (deg)
    for a given (Cap, Vol, TRS, Feed).
    """
    x_dict = {
        'Capacitance (pF)': cap,
        'Voltage (V)': vol,
        'TRS (rpm)': trs,
        'Feed rate (µm/s)': feed,
    }
    x_row = pd.DataFrame([x_dict])[input_features]

    x_scaled = scaler_X.transform(x_row)
    pred_scaled = {}
    for out in output_features:
        model = models[out]
        pred_scaled[out] = model.predict(x_scaled)[0]
    pred_scaled_df = pd.DataFrame([pred_scaled])
    pred_unscaled = scaler_y.inverse_transform(pred_scaled_df)[0]

    pred_entry = float(pred_unscaled[output_features.index('Entry dia')])
    pred_exit  = float(pred_unscaled[output_features.index('Exit dia')])

    taper_rad = compute_taper_angle(pred_entry, pred_exit, depth=HOLE_DEPTH)
    taper_deg = float(np.degrees(taper_rad))

    return pred_entry, pred_exit, taper_deg

# ----------------------------------------------------------------------
# 4. Evaluate ALL settings once (global process space)
# ----------------------------------------------------------------------

def evaluate_full_design_space() -> pd.DataFrame:
    """
    Evaluate surrogate at all combinations of Cap, Vol, TRS, Feed
    and return a DataFrame with geometry and taper.
    """
    cap_vals = DEFAULT_LEVELS['Capacitance (pF)']
    vol_vals = DEFAULT_LEVELS['Voltage (V)']
    trs_vals = DEFAULT_LEVELS['TRS (rpm)']
    feed_vals = DEFAULT_LEVELS['Feed rate (µm/s)']

    rows = []
    for cap, vol, trs, feed in itertools.product(cap_vals, vol_vals, trs_vals, feed_vals):
        pred_entry, pred_exit, taper_deg = predict_geometry(cap, vol, trs, feed)
        rows.append({
            'Capacitance (pF)': cap,
            'Voltage (V)': vol,
            'TRS (rpm)': trs,
            'Feed rate (µm/s)': feed,
            'Pred_Entry': pred_entry,
            'Pred_Exit': pred_exit,
            'Pred_Taper': taper_deg,
        })

    return pd.DataFrame(rows)

def print_objective_definition():
    """
    Pretty-print the inverse-design objective definition and normalization scales.
    """
    print("\n=== Inverse-design objective definition ===")
    print("f1(x) = ((Den(x) - Den_target) / den_std)^2"
          "       + ((Dex(x) - Dex_target) / dex_std)^2")
    print("f2(x) = (Taper_deg(x) / taper_std)^2")
    print("J(x)  = f1(x) + lambda * f2(x)")
    print(f"\nCurrent normalization constants:")
    print(f"  den_std  = {den_std:.3f} µm")
    print(f"  dex_std  = {dex_std:.3f} µm")

# ----------------------------------------------------------------------
# 5. Inverse design: normalized diameter + normalized taper
# ----------------------------------------------------------------------

def inverse_design(
    target_entry: float,
    target_exit: float,
    df_all: pd.DataFrame | None = None,
    taper_max: float | None = None,
    taper_weight: float = 2,
    top_k: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Given a target (Entry, Exit), search over df_all (pre-evaluated full space)
    and minimize:

        f1 = (normalized Entry error)^2 + (normalized Exit error)^2
        f2 = (Pred_Taper / TAPER_REF_DEG)^2
        Obj = f1 + taper_weight * f2

    If taper_max is not None, discard rows with Pred_Taper > taper_max.
    If df_all is None, this function will evaluate the full space internally.
    """
    if df_all is None:
        df_all = evaluate_full_design_space()

    df = df_all.copy()

    if taper_max is not None:
        df = df[df['Pred_Taper'] <= taper_max]

    if df.empty:
        raise ValueError("No feasible candidates under taper_max constraint.")

    # f1: normalized squared diameter error
    err_entry = (df['Pred_Entry'] - target_entry) / den_std
    err_exit  = (df['Pred_Exit']  - target_exit)  / dex_std

    f1 = err_entry**2 + err_exit**2
    taper_std = float(df_all['Pred_Taper'].std(ddof=1))

    # f2: normalized squared taper using taper_std
    norm_taper = df['Pred_Taper'] / taper_std
    f2 = norm_taper**2
    
    if verbose:
        print("\n--- Normalization and objective statistics ---")
        print(f"den_std = {den_std:.3f} µm, dex_std = {dex_std:.3f} µm, "
              f"taper_std = {taper_std:.3f} deg")
        print(f"f1 range over candidates: [{f1.min():.3f}, {f1.max():.3f}]")
        print(f"f2 range over candidates: [{f2.min():.3f}, {f2.max():.3f}]")
        print(f"(These are before multiplying f2 by lambda = {taper_weight})")


    obj = f1 + taper_weight * f2

    df = df.assign(
        EntryErrNorm=err_entry,
        ExitErrNorm=err_exit,
        F1_Diameter=f1,
        F2_Taper=f2,
        ObjValue=obj,
    )

    df_sorted = df.sort_values('ObjValue', ascending=True, ignore_index=True)

    return df_sorted.head(top_k)


def inverse_design_entry_only(
    target_entry: float,
    df_all: pd.DataFrame | None = None,
    taper_max: float | None = None,
    taper_weight: float = 0,
    top_k: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    if df_all is None:
        df_all = evaluate_full_design_space()
    df = df_all.copy()
    if taper_max is not None:
        df = df[df["Pred_taper"] <= taper_max]
    if df.empty:
        raise ValueError("No Feasible candidates")

    err_entry = (df["Pred_Entry"] - target_entry) / den_std
    f1_entry = err_entry**2

    taper_std = float(df_all["Pred_Taper"].std(ddof=1))
    norm_taper = df["Pred_Taper"] / taper_std
    f2_taper = norm_taper**2
    if verbose:
        print("\n--- Entry-Only Inverse-Design Statistics ---")
        print(f"den_std = {den_std:.4f} µm, taper_std = {taper_std:.4f} deg")
        print(f"f1_entry range: [{f1_entry.min():.4f}, {f1_entry.max():.4f}]")
        print(f"f1_entry range: [{f2_taper.min():.4f}, {f2_taper.max():.4f}]")
        print(f"(Before multiplying f2_taper by lambda = {taper_weight})")

    obj = f1_entry + taper_weight * f2_taper
    df = df.assign(
        EntryErrNorm=err_entry,
        F1_Entry=f1_entry,
        F2_Taper=f2_taper,
        ObjValue=obj,)
    df_sorted = df.sort_values("ObjValue", ascending = True, ignore_index = True)
    return df_sorted.head(top_k)
    
# ----------------------------------------------------------------------
# 6. Experimental lookup for validation
# ----------------------------------------------------------------------

def lookup_experimental_match(row: pd.Series, df_exp: pd.DataFrame) -> pd.Series | None:
  
    mask = (
        (df_exp['Capacitance (pF)'] == row['Capacitance (pF)']) &
        (df_exp['Voltage (V)'] == row['Voltage (V)']) &
        (df_exp['TRS (rpm)'] == row['TRS (rpm)']) &
        (df_exp['Feed rate (µm/s)'] == row['Feed rate (µm/s)'])
    )
    matches = df_exp[mask]
    if matches.empty:
        return None
    return matches.iloc[0]

# ----------------------------------------------------------------------
# 7. Local sensitivity: neighborhood in Cap/Vol for a chosen setting
# ----------------------------------------------------------------------

def local_sensitivity_cap_vol(base_row: pd.Series) -> pd.DataFrame:
    """
    For a chosen setting, evaluate neighboring Cap/Vol levels (±1)
    with TRS and Feed fixed, and return their geometry and taper.
    """
    cap_vals = DEFAULT_LEVELS['Capacitance (pF)']
    vol_vals = DEFAULT_LEVELS['Voltage (V)']

    base_cap = base_row['Capacitance (pF)']
    base_vol = base_row['Voltage (V)']
    trs = base_row['TRS (rpm)']
    feed = base_row['Feed rate (µm/s)']

    cap_index = cap_vals.index(base_cap)
    vol_index = vol_vals.index(base_vol)

    neighbor_caps = [cap_vals[i] for i in range(
        max(0, cap_index - 1),
        min(len(cap_vals), cap_index + 2)
    )]
    neighbor_vols = [vol_vals[j] for j in range(
        max(0, vol_index - 1),
        min(len(vol_vals), vol_index + 2)
    )]

    rows = []
    for nc, nv in itertools.product(neighbor_caps, neighbor_vols):
        pred_entry, pred_exit, taper_deg = predict_geometry(nc, nv, trs, feed)
        rows.append({
            'Capacitance (pF)': nc,
            'Voltage (V)': nv,
            'TRS (rpm)': trs,
            'Feed rate (µm/s)': feed,
            'Pred_Entry': pred_entry,
            'Pred_Exit': pred_exit,
            'Pred_Taper': taper_deg,
        })

    return pd.DataFrame(rows).sort_values("Pred_Taper", ignore_index=True)

def lambda_sweep(target_entry: float, target_exit: float, df_all: pd.DataFrame, lambdas=(0.0, 0.5, 1.0, 1.5,2,2.5,3,3.5,4)):
    """
    For a fixed target, run inverse design for several taper_weight values
    and print how taper and diameter error change for the best solution.
    """
    rows = []
    for lam in lambdas:
        inv = inverse_design(
            target_entry=target_entry,
            target_exit=target_exit,
            df_all=df_all,
            taper_max=None,
            taper_weight=lam,
            top_k=1,
        )
        r = inv.iloc[0]
        rows.append({
            "lambda": lam,
            "Pred_Entry": r["Pred_Entry"],
            "Pred_Exit": r["Pred_Exit"],
            "Pred_Taper": r["Pred_Taper"],
            "F1_Diameter": r["F1_Diameter"],
            "F2_Taper": r["F2_Taper"],
            "ObjValue": r["ObjValue"],
        })
    sweep_df = pd.DataFrame(rows)
    print("\n=== Lambda sweep results ===")
    print(sweep_df)
    return sweep_df

# ----------------------------------------------------------------------
# 8. Visualization helpers
# ----------------------------------------------------------------------

def scatter_process_vs_taper(df_all: pd.DataFrame):
    """
    Scatter of Voltage vs Capacitance coloured by taper (all TRS/Feed).
    """
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all['Voltage (V)'],
        df_all['Capacitance (pF)'],
        c=df_all['Pred_Taper'],
        cmap='magma_r',
        s=40,
        edgecolor='none'
    )
    plt.xlabel("Voltage (V)")
    plt.ylabel("Capacitance (pF)")
    plt.title("Taper (deg) across all process settings")
    cbar = plt.colorbar(sc)
    cbar.set_label("Taper (deg)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_voltage_cap_taper.png"), dpi=600)
    plt.close()

def scatter_geometry_with_inverse_points(df_all: pd.DataFrame, inv_results: pd.DataFrame,tag: str = ""):
    """
    Entry vs Exit scatter (all settings) coloured by taper,
    with inverse-designed top-k points overlaid.
    """
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all['Pred_Entry'],
        df_all['Pred_Exit'],
        c=df_all['Pred_Taper'],
        cmap='viridis',
        s=30,
        alpha=0.5,
        edgecolor='none'
    )
    plt.xlabel("Predicted Entry dia")
    plt.ylabel("Predicted Exit dia")
    plt.title("Geometry cloud with inverse-designed solutions")
    cbar = plt.colorbar(sc)
    cbar.set_label("Taper (deg)")

    plt.scatter(
        inv_results['Pred_Entry'],
        inv_results['Pred_Exit'],
        marker='o',
        color='red',
        edgecolor='black',
        s=80,
        label='Inverse-design top-k'
    )
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_Geometry_with_inverse_points"), dpi=600)
    plt.close()

def scatter_geometry_cloud(df_all: pd.DataFrame, tag: str = ""):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all['Pred_Entry'],
        df_all['Pred_Exit'],
        c=df_all['Pred_Taper'],
        cmap='viridis',
        s=30,
        alpha=0.5,
        edgecolor='none'
    )
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.title("Geometry cloud (Entry vs Exit, coloured by taper)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Taper (deg)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"scAtter_geometry_cloud{('_' + tag) if tag else ''}.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=600)
    plt.close()


def plot_local_taper_heatmap(df_neighbors: pd.DataFrame, base_row: pd.Series):
    """
    Local heatmap of taper over neighboring Cap/Vol around a base setting.
    """
    cap_vals = sorted(df_neighbors['Capacitance (pF)'].unique())
    vol_vals = sorted(df_neighbors['Voltage (V)'].unique())

    grid = np.zeros((len(cap_vals), len(vol_vals)))
    for i, cap in enumerate(cap_vals):
        for j, vol in enumerate(vol_vals):
            val = df_neighbors[
                (df_neighbors['Capacitance (pF)'] == cap) &
                (df_neighbors['Voltage (V)'] == vol)
            ]['Pred_Taper'].values[0]
            grid[i, j] = val

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        grid,
        xticklabels=vol_vals,
        yticklabels=cap_vals,
        cmap="magma_r",
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Capacitance (pF)")
    ax.set_title(
        f"Local taper (deg) around Cap={base_row['Capacitance (pF)']}, "
        f"Vol={base_row['Voltage (V)']}"
    )

    base_cap = base_row['Capacitance (pF)']
    base_vol = base_row['Voltage (V)']
    i_base = cap_vals.index(base_cap)
    j_base = vol_vals.index(base_vol)
    plt.scatter(j_base + 0.5, i_base + 0.5, marker='o',
                color='cyan', edgecolor='black', s=80, label='Base setting')
    plt.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_local_taper.png"), dpi=600)
    plt.close()

# MAIN

if __name__ == "__main__":

    df_all = evaluate_full_design_space()
    print_objective_definition()

    scatter_process_vs_taper(df_all)

    targets = [
        (590.36, 588.94),  
        (580.0, 540.0),    
        (620.0, 580.0),   
    ]

    all_inv_results = []

    for i, (target_entry, target_exit) in enumerate(targets):
        print(f"\n=== Inverse design for target Entry={target_entry}, Exit={target_exit} ===")
        inv_results = inverse_design(
            target_entry=target_entry,
            target_exit=target_exit,
            df_all=df_all,
            taper_max=None,      
            taper_weight=2,    
            top_k=5,
            verbose=(i == 0),
        )
        print(inv_results)
        all_inv_results.append(inv_results)

        scatter_geometry_with_inverse_points(
            df_all, inv_results.head(1),
            tag=f"{int(target_entry)}_{int(target_exit)}"
        )
        scatter_geometry_cloud(df_all, tag="full_space")
        
        best_row = inv_results.iloc[0]
        exp_match = lookup_experimental_match(best_row, df)
        if exp_match is not None:
            print("\nMatching experimental row for best solution:")
            print(exp_match[['Capacitance (pF)', 'Voltage (V)', 'TRS (rpm)',
                             'Feed rate (µm/s)', 'Entry dia', 'Exit dia']])
        else:
            print("\nNo exact experimental match found for this setting in Final.csv.")


    
    # index 1  -> targets[1] = (580.0, 540.0)
    base_row = all_inv_results[1].iloc[0]   
    
    print("\n=== Local sensitivity around best inverse-designed setting (580, 540) ===")
    print(base_row)
    
    df_neighbors = local_sensitivity_cap_vol(base_row)
    print("\nLocal neighborhood (sorted by Pred_Taper):")
    print(df_neighbors)
    
    plot_local_taper_heatmap(df_neighbors, base_row)



    for (target_entry, target_exit), inv_df in zip(targets, all_inv_results):
        fname = f"inverse_results_target_{int(target_entry)}_{int(target_exit)}.csv"
        fpath = os.path.join(RESULTS_DIR, fname)
        inv_df.to_csv(fpath, index=False)
        print(f"Saved inverse-design results to {fpath}")

    # Lambda sweep CSV for one representative target
    sweep_df = lambda_sweep(target_entry=590.36, target_exit=588.94, df_all=df_all)
    sweep_path = os.path.join(RESULTS_DIR, "lambda_sweep_590.36_588.94.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved lambda sweep to {sweep_path}")

        


    den_min, den_max = float(df['Entry dia'].min()), float(df['Entry dia'].max())
    dex_min, dex_max = float(df['Exit dia'].min()), float(df['Exit dia'].max())

    candidate_targets = []

    candidate_targets.append({"Target_ID": "T1", "Target_Entry": 580.0, "Target_Exit": 540.0})
    candidate_targets.append({"Target_ID": "T2", "Target_Entry": 600.0, "Target_Exit": 560.0})
    candidate_targets.append({"Target_ID": "T3", "Target_Entry": 620.0, "Target_Exit": 580.0})


    for k in range(4, 11):
        alpha = (k - 4) / 6.0
        den = den_min + alpha * (den_max - den_min)
        dex = dex_min + alpha * (dex_max - dex_min)
        candidate_targets.append({
            "Target_ID": f"T{k}",
            "Target_Entry": round(den, 2),
            "Target_Exit": round(dex, 2),
        })

    candidate_targets_df = pd.DataFrame(candidate_targets)
    targets_path = os.path.join(RESULTS_DIR, "candidate_targets_10.csv")
    candidate_targets_df.to_csv(targets_path, index=False)
    print(f"\nSaved 10 candidate target geometries to {targets_path}")

    print(candidate_targets_df)
        
    target_entry_single = 630
    inv_entry_only = inverse_design_entry_only(
        target_entry=target_entry_single,
        df_all=df_all,
        taper_max=None,
        taper_weight=0.0,
        top_k=5,          
        verbose=True,
    )
    

    inv_entry_only_sorted = inv_entry_only.sort_values("ObjValue", ascending=True)
    
    entry_only_path = os.path.join(RESULTS_DIR, f"entry_only_inverse_{int(target_entry_single)}.csv")
    inv_entry_only_sorted.to_csv(entry_only_path, index=False)
    print(f"\nSaved entry-only inverse-design (sorted by taper) to {entry_only_path}")
    

    plt.figure(figsize=(6, 5))
    plt.scatter(
        inv_entry_only_sorted["Pred_Entry"],
        inv_entry_only_sorted["Pred_Taper"],
        c=inv_entry_only_sorted["Pred_Taper"],
        cmap="magma_r",
        s=60,
        edgecolor="black",
    )
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Taper (deg)")
    plt.title(f"Entry-only inverse design (target Den = {target_entry_single} µm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOT_DIR, f"entry_only_inverse_{int(target_entry_single)}.png")
    plt.savefig(plot_path, dpi=600)   
    plt.close()
    
    print(f"Saved entry-only inverse-design plot to {plot_path}")
