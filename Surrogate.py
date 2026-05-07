"""
Surrogate-based inverse design and taper analysis for micro-EDM.

UPDATED: Added Cap–Voltage validity constraints based on experimental design table.
Valid combinations:
  Cap 100 pF   → Voltage: 180
  Cap 1000 pF  → Voltage: 145, 180
  Cap 10000 pF → Voltage: 75, 110, 145, 180

All design-space enumeration, inverse-design search, and local
sensitivity functions now filter out infeasible (Cap, Voltage) pairs
before any surrogate calls.
"""

import os
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("whitegrid")

PLOT_DIR = "INVERSE_plots"
os.makedirs(PLOT_DIR, exist_ok=True)
RESULTS_DIR = "INVERSE_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Paths and global config
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH      = "Final.csv"
SURROGATE_PATH = os.path.join("xgboost_multi_output_model",
                              "xgboost_surrogate_seed0.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Cap–Voltage validity table  ← NEW
# ─────────────────────────────────────────────────────────────────────────────

# Directly transcribed from Table 2 (Variable parameters section).
# Keys are Capacitance values (pF); values are the allowed Voltage levels (V).
VALID_CAP_VOL: dict[float, list[float]] = {
    100.0:   [180.0],
    1000.0:  [145.0, 180.0],
    10000.0: [75.0, 110.0, 145.0, 180.0],
}

# Pre-build a frozenset of valid (cap, vol) tuples for O(1) lookup
VALID_CAP_VOL_PAIRS: frozenset[tuple[float, float]] = frozenset(
    (cap, vol)
    for cap, vols in VALID_CAP_VOL.items()
    for vol in vols
)


def is_valid_cap_vol(cap: float, vol: float) -> bool:
    """Return True iff (cap, vol) is a permitted combination."""
    return (float(cap), float(vol)) in VALID_CAP_VOL_PAIRS


def validate_setting(cap: float, vol: float, trs: float = None, feed: float = None,
                     raise_on_invalid: bool = False) -> bool:
    """
    Check a full or partial process setting for Cap–Vol compatibility.
    Prints a warning (or raises) if the combination is not in the design table.
    """
    valid = is_valid_cap_vol(cap, vol)
    if not valid:
        msg = (
            f"  [INVALID SETTING] Cap={cap} pF + Voltage={vol} V is not a permitted "
            f"combination.\n"
            f"  Allowed voltages for Cap={cap} pF: "
            f"{VALID_CAP_VOL.get(float(cap), 'unknown cap level')}"
        )
        if raise_on_invalid:
            raise ValueError(msg)
        print(msg)
    return valid


def filter_valid_cap_vol_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows from a DataFrame whose (Cap, Vol) pair is not in the design table.
    Prints a summary of how many rows were removed.
    """
    mask = df.apply(
        lambda r: is_valid_cap_vol(r["Capacitance (pF)"], r["Voltage (V)"]), axis=1
    )
    n_removed = (~mask).sum()
    if n_removed:
        print(f"  [Constraint filter] Removed {n_removed} infeasible (Cap, Vol) rows "
              f"({len(df)} → {mask.sum()}).")
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load data and surrogate bundle
# ─────────────────────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATA_PATH)

columns_to_drop = [
    "Sl No. (Image)", "Unnamed: 6",
    "OC at Entry", "OC at Exit", "MRR", "Taper(degree)",
]
df = df_raw.drop(columns=columns_to_drop, errors="ignore")
# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL NORMALIZATION CONSTANTS (computed BEFORE constraint filtering)
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_DEN_STD = float(df_raw["Entry dia"].std(ddof=1))
GLOBAL_DEX_STD = float(df_raw["Exit dia"].std(ddof=1))

print(f"Den range: {df['Entry dia'].min():.2f}–{df['Entry dia'].max():.2f} µm")
print(f"Dex range: {df['Exit dia'].min():.2f}–{df['Exit dia'].max():.2f} µm")

# Validate raw data itself against the constraint table
print("\n[Data audit] Checking experimental data for invalid (Cap, Vol) combos…")
df = filter_valid_cap_vol_df(df)

bundle       = joblib.load(SURROGATE_PATH)
models       = bundle["models"]
scaler_X     = bundle["scaler_X"]
scaler_y     = bundle["scaler_y"]
input_features  = bundle["input_features"]
output_features = bundle["output_features"]

assert "Entry dia" in output_features and "Exit dia" in output_features, \
    "Surrogate bundle must contain Entry dia and Exit dia models."



den_std = GLOBAL_DEN_STD
dex_std = GLOBAL_DEX_STD

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Discrete design space helpers
# ─────────────────────────────────────────────────────────────────────────────

HOLE_DEPTH = 700.0  # µm


def get_default_levels(df: pd.DataFrame) -> dict:
    levels = {}
    for col in ["Capacitance (pF)", "Voltage (V)", "TRS (rpm)", "Feed rate (µm/s)"]:
        levels[col] = sorted(df[col].unique())
    return levels


DEFAULT_LEVELS = get_default_levels(df)


def compute_taper_angle(entry_d: float, exit_d: float,
                        depth: float = HOLE_DEPTH) -> float:
    return np.arctan((entry_d - exit_d) / (2.0 * depth))


def valid_cap_vol_combinations() -> list[tuple[float, float]]:
    """Return all (cap, vol) pairs that satisfy the design table."""
    return sorted(VALID_CAP_VOL_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Core evaluator
# ─────────────────────────────────────────────────────────────────────────────

def predict_geometry(cap: float, vol: float,
                     trs: float, feed: float) -> tuple[float, float, float]:
    """
    Predict Entry, Exit and Taper for a given setting.
    Raises ValueError if (cap, vol) is an invalid combination.
    """
    validate_setting(cap, vol, trs, feed, raise_on_invalid=True)

    x_dict = {
        "Capacitance (pF)":  cap,
        "Voltage (V)":       vol,
        "TRS (rpm)":         trs,
        "Feed rate (µm/s)":  feed,
    }
    x_row    = pd.DataFrame([x_dict])[input_features]
    x_scaled = scaler_X.transform(x_row)

    pred_scaled = {out: models[out].predict(x_scaled)[0] for out in output_features}
    pred_scaled_df = pd.DataFrame([pred_scaled])
    pred_unscaled  = scaler_y.inverse_transform(pred_scaled_df)[0]

    pred_entry = float(pred_unscaled[output_features.index("Entry dia")])
    pred_exit  = float(pred_unscaled[output_features.index("Exit dia")])
    taper_deg  = float(np.degrees(compute_taper_angle(pred_entry, pred_exit)))

    return pred_entry, pred_exit, taper_deg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full design-space evaluation (constrained)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full_design_space() -> pd.DataFrame:
    """
    Evaluate surrogate at every VALID (Cap, Vol) × TRS × Feed combination.
    Invalid (Cap, Vol) pairs are skipped entirely.
    """
    trs_vals  = DEFAULT_LEVELS["TRS (rpm)"]
    feed_vals = DEFAULT_LEVELS["Feed rate (µm/s)"]
    cap_vol_pairs = valid_cap_vol_combinations()

    rows = []
    n_skipped = 0
    for (cap, vol), trs, feed in itertools.product(cap_vol_pairs, trs_vals, feed_vals):
        # Double-check (should always pass here, but keeps logic explicit)
        if not is_valid_cap_vol(cap, vol):
            n_skipped += 1
            continue
        pred_entry, pred_exit, taper_deg = predict_geometry(cap, vol, trs, feed)
        rows.append({
            "Capacitance (pF)":  cap,
            "Voltage (V)":       vol,
            "TRS (rpm)":         trs,
            "Feed rate (µm/s)":  feed,
            "Pred_Entry":        pred_entry,
            "Pred_Exit":         pred_exit,
            "Pred_Taper":        taper_deg,
        })

    print(f"  [Design space] Evaluated {len(rows)} valid settings "
          f"(skipped {n_skipped} invalid Cap–Vol pairs).")
    return pd.DataFrame(rows)
# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL TAPER NORMALIZATION (computed once from constrained space)
# ─────────────────────────────────────────────────────────────────────────────

df_all = evaluate_full_design_space()
GLOBAL_TAPER_STD = float(df_all["Pred_Taper"].std(ddof=1))
print(f"[GLOBAL NORMALIZATION] taper_std = {GLOBAL_TAPER_STD:.4f}")

def print_objective_definition():
    print("\n=== Inverse-design objective definition ===")
    print("f1(x) = ((Den(x) - Den_target) / den_std)^2"
          "       + ((Dex(x) - Dex_target) / dex_std)^2")
    print("f2(x) = (Taper_deg(x) / taper_std)^2")
    print("J(x)  = f1(x) + lambda * f2(x)")
    print(f"\nNormalization constants:")
    print(f"  den_std  = {den_std:.3f} µm")
    print(f"  dex_std  = {dex_std:.3f} µm")
    print(f"  taper_std = {GLOBAL_TAPER_STD:.3f} deg")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Inverse design
# ─────────────────────────────────────────────────────────────────────────────

def inverse_design(
    target_entry: float,
    target_exit:  float,
    df_all:       pd.DataFrame | None = None,
    taper_max:    float | None = None,
    taper_weight: float = 1.0,
    top_k:        int   = 5,
    verbose:      bool  = False,
) -> pd.DataFrame:
    """
    Search the pre-evaluated constrained design space for the settings that
    minimise J = f1 + taper_weight * f2.
    All rows in df_all are already guaranteed to be valid (Cap, Vol) pairs.
    """
    if df_all is None:
        df_all = evaluate_full_design_space()

    df = df_all.copy()

    if taper_max is not None:
        df = df[df["Pred_Taper"] <= taper_max]
        if df.empty:
            raise ValueError("No feasible candidates under taper_max constraint.")

    err_entry = (df["Pred_Entry"] - target_entry) / den_std
    err_exit  = (df["Pred_Exit"]  - target_exit)  / dex_std
    f1 = err_entry**2 + err_exit**2

    taper_std = GLOBAL_TAPER_STD
    norm_taper = df["Pred_Taper"] / taper_std
    f2 = norm_taper**2

    if verbose:
        print("\n--- Normalization and objective statistics ---")
        print(f"den_std={den_std:.3f} µm, dex_std={dex_std:.3f} µm, "
              f"taper_std={taper_std:.3f} deg")
        print(f"f1 range: [{f1.min():.3f}, {f1.max():.3f}]")
        print(f"f2 range: [{f2.min():.3f}, {f2.max():.3f}]")
        print(f"(f2 multiplied by lambda = {taper_weight})")

    obj = f1 + taper_weight * f2
    df  = df.assign(
        EntryErrNorm=err_entry,
        ExitErrNorm=err_exit,
        F1_Diameter=f1,
        F2_Taper=f2,
        ObjValue=obj,
    )
    return df.sort_values("ObjValue", ascending=True, ignore_index=True).head(top_k)


def inverse_design_entry_only(
    target_entry: float,
    df_all:       pd.DataFrame | None = None,
    taper_max:    float | None = None,
    taper_weight: float = 0.0,
    top_k:        int   = 5,
    verbose:      bool  = False,
) -> pd.DataFrame:
    if df_all is None:
        df_all = evaluate_full_design_space()

    df = df_all.copy()
    if taper_max is not None:
        df = df[df["Pred_Taper"] <= taper_max]
        if df.empty:
            raise ValueError("No feasible candidates.")

    err_entry = (df["Pred_Entry"] - target_entry) / den_std
    f1_entry  = err_entry**2

    taper_std = GLOBAL_TAPER_STD
    norm_taper  = df["Pred_Taper"] / taper_std
    f2_taper    = norm_taper**2

    if verbose:
        print("\n--- Entry-Only Inverse-Design Statistics ---")
        print(f"den_std={den_std:.4f} µm, taper_std={taper_std:.4f} deg")
        print(f"f1_entry range: [{f1_entry.min():.4f}, {f1_entry.max():.4f}]")
        print(f"f2_taper range: [{f2_taper.min():.4f}, {f2_taper.max():.4f}]")

    obj = f1_entry + taper_weight * f2_taper
    df  = df.assign(
        EntryErrNorm=err_entry,
        F1_Entry=f1_entry,
        F2_Taper=f2_taper,
        ObjValue=obj,
    )
    return df.sort_values("ObjValue", ascending=True, ignore_index=True).head(top_k)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Experimental lookup
# ─────────────────────────────────────────────────────────────────────────────

def lookup_experimental_match(row: pd.Series, df_exp: pd.DataFrame) -> pd.Series | None:
    mask = (
        (df_exp["Capacitance (pF)"] == row["Capacitance (pF)"]) &
        (df_exp["Voltage (V)"]      == row["Voltage (V)"]) &
        (df_exp["TRS (rpm)"]        == row["TRS (rpm)"]) &
        (df_exp["Feed rate (µm/s)"] == row["Feed rate (µm/s)"])
    )
    matches = df_exp[mask]
    return matches.iloc[0] if not matches.empty else None


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Local sensitivity (Cap/Vol neighbourhood, constrained)
# ─────────────────────────────────────────────────────────────────────────────

def local_sensitivity_cap_vol(base_row: pd.Series) -> pd.DataFrame:
    """
    Evaluate neighbouring Cap/Vol levels (±1 step) around a base setting,
    skipping any (Cap, Vol) pair not in the design table.
    """
    cap_vals = DEFAULT_LEVELS["Capacitance (pF)"]
    vol_vals = DEFAULT_LEVELS["Voltage (V)"]

    base_cap = base_row["Capacitance (pF)"]
    base_vol = base_row["Voltage (V)"]
    trs      = base_row["TRS (rpm)"]
    feed     = base_row["Feed rate (µm/s)"]

    cap_idx = cap_vals.index(base_cap)
    vol_idx = vol_vals.index(base_vol)

    neighbor_caps = [cap_vals[i] for i in range(
        max(0, cap_idx - 1), min(len(cap_vals), cap_idx + 2))]
    neighbor_vols = [vol_vals[j] for j in range(
        max(0, vol_idx - 1), min(len(vol_vals), vol_idx + 2))]

    rows = []
    skipped = []
    for nc, nv in itertools.product(neighbor_caps, neighbor_vols):
        if not is_valid_cap_vol(nc, nv):
            skipped.append((nc, nv))
            continue
        pred_entry, pred_exit, taper_deg = predict_geometry(nc, nv, trs, feed)
        rows.append({
            "Capacitance (pF)":  nc,
            "Voltage (V)":       nv,
            "TRS (rpm)":         trs,
            "Feed rate (µm/s)":  feed,
            "Pred_Entry":        pred_entry,
            "Pred_Exit":         pred_exit,
            "Pred_Taper":        taper_deg,
        })

    if skipped:
        print(f"  [Local sensitivity] Skipped {len(skipped)} invalid (Cap, Vol) neighbours: "
              + ", ".join(f"({c},{v})" for c, v in skipped))

    return pd.DataFrame(rows).sort_values("Pred_Taper", ignore_index=True)


def lambda_sweep(target_entry: float, target_exit: float,
                 df_all: pd.DataFrame,
                 lambdas=(0.0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4)):
    rows = []
    for lam in lambdas:
        inv = inverse_design(target_entry=target_entry, target_exit=target_exit,
                             df_all=df_all, taper_max=None,
                             taper_weight=lam, top_k=3)
        r = inv.iloc[0]
        rows.append({
            "lambda":      lam,
            "Pred_Entry":  r["Pred_Entry"],
            "Pred_Exit":   r["Pred_Exit"],
            "Pred_Taper":  r["Pred_Taper"],
            "F1_Diameter": r["F1_Diameter"],
            "F2_Taper":    r["F2_Taper"],
            "ObjValue":    r["ObjValue"],
        })
    sweep_df = pd.DataFrame(rows)
    print("\n=== Lambda sweep results ===")
    print(sweep_df)
    return sweep_df


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def scatter_process_vs_taper(df_all: pd.DataFrame):
    """Voltage vs Capacitance coloured by taper.
    Only valid (Cap, Vol) pairs appear (design space is already filtered)."""
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all["Voltage (V)"], df_all["Capacitance (pF)"],
        c=df_all["Pred_Taper"], cmap="magma_r", s=40, edgecolor="none"
    )
    plt.xlabel("Voltage (V)")
    plt.ylabel("Capacitance (pF)")
    plt.title("Taper (deg) across valid process settings")
    plt.colorbar(sc, label="Taper (deg)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_voltage_cap_taper.png"), dpi=600)
    plt.close()


def scatter_geometry_with_inverse_points(df_all: pd.DataFrame,
                                         inv_results: pd.DataFrame,
                                         tag: str = ""):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all["Pred_Entry"], df_all["Pred_Exit"],
        c=df_all["Pred_Taper"], cmap="viridis",
        s=30, alpha=0.5, edgecolor="none"
    )
    plt.colorbar(sc, label="Taper (deg)")
    plt.scatter(
        inv_results["Pred_Entry"], inv_results["Pred_Exit"],
        marker="o", color="red", edgecolor="black", s=80,
        label="Inverse-design top-k"
    )
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.title("Geometry cloud with inverse-designed solutions")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"scatter_geometry_inv_{tag}.png" if tag else "scatter_Geometry_with_inverse_points.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=600)
    plt.close()


def scatter_geometry_cloud(df_all: pd.DataFrame, tag: str = ""):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        df_all["Pred_Entry"], df_all["Pred_Exit"],
        c=df_all["Pred_Taper"], cmap="viridis",
        s=30, alpha=0.5, edgecolor="none"
    )
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.title("Geometry cloud (Entry vs Exit, coloured by taper)")
    plt.colorbar(sc, label="Taper (deg)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"scatter_geometry_cloud{'_' + tag if tag else ''}.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=600)
    plt.close()


def plot_local_taper_heatmap(df_neighbors: pd.DataFrame, base_row: pd.Series):
    """
    Local taper heatmap.  Cells for invalid (Cap, Vol) pairs are masked out
    so they are visually distinct (grey) rather than silently missing.
    """
    cap_vals = sorted(df_neighbors["Capacitance (pF)"].unique())
    vol_vals = sorted(df_neighbors["Voltage (V)"].unique())

    grid = np.full((len(cap_vals), len(vol_vals)), np.nan)
    for i, cap in enumerate(cap_vals):
        for j, vol in enumerate(vol_vals):
            sub = df_neighbors[
                (df_neighbors["Capacitance (pF)"] == cap) &
                (df_neighbors["Voltage (V)"] == vol)
            ]
            if not sub.empty:
                grid[i, j] = sub["Pred_Taper"].values[0]

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(
        grid,
        xticklabels=vol_vals,
        yticklabels=cap_vals,
        cmap="magma_r",
        annot=True,
        fmt=".2f",
        mask=np.isnan(grid),   # grey-out invalid cells
        cbar=True,
    )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Capacitance (pF)")
    ax.set_title(
        f"Local taper (deg) around Cap={base_row['Capacitance (pF)']}, "
        f"Vol={base_row['Voltage (V)']}"
    )

    base_cap = base_row["Capacitance (pF)"]
    base_vol = base_row["Voltage (V)"]
    i_base = cap_vals.index(base_cap)
    j_base = vol_vals.index(base_vol)
    ax.scatter(j_base + 0.5, i_base + 0.5, marker="o",
               color="cyan", edgecolor="black", s=80, label="Base setting", zorder=5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "scatter_local_taper.png"), dpi=600)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  NEW — Cap–Voltage constraint validity map
# ─────────────────────────────────────────────────────────────────────────────

def plot_cap_vol_validity_map():
    """
    Heatmap showing which (Cap, Vol) combinations are valid (green)
    vs invalid / not in design table (red).
    Useful for reporting and sanity-checking constraint enforcement.
    """
    cap_vals = sorted(VALID_CAP_VOL.keys())
    vol_vals = sorted({v for vlist in VALID_CAP_VOL.values() for v in vlist})

    grid = np.zeros((len(cap_vals), len(vol_vals)), dtype=int)
    for i, cap in enumerate(cap_vals):
        for j, vol in enumerate(vol_vals):
            grid[i, j] = int(is_valid_cap_vol(cap, vol))

    fig, ax = plt.subplots(figsize=(6, 3))
    cmap = plt.cm.colors.ListedColormap(["#e74c3c", "#2ecc71"])
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(vol_vals)))
    ax.set_yticks(range(len(cap_vals)))
    ax.set_xticklabels([f"{v:.0f} V" for v in vol_vals])
    ax.set_yticklabels([f"{c:.0f} pF" for c in cap_vals])
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Capacitance (pF)")
    ax.set_title("Cap–Voltage Validity Map\n(green = allowed, red = excluded)")

    for i in range(len(cap_vals)):
        for j in range(len(vol_vals)):
            label = "✓" if grid[i, j] else "✗"
            ax.text(j, i, label, ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")

    patches = [
        mpatches.Patch(color="#2ecc71", label="Valid combination"),
        mpatches.Patch(color="#e74c3c", label="Excluded combination"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cap_vol_validity_map.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved Cap–Vol validity map to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  NEW — Taper breakdown by valid Cap–Vol group
# ─────────────────────────────────────────────────────────────────────────────

def plot_taper_by_cap_vol_group(df_all: pd.DataFrame):
    """
    Box + strip plot of predicted taper grouped by (Cap, Vol) pair.
    Illustrates how the constraint-filtered design space distributes
    across the nine valid parameter groups.
    """
    df_plot = df_all.copy()
    df_plot["Cap–Vol group"] = (
        df_plot["Capacitance (pF)"].astype(int).astype(str) + " pF / "
        + df_plot["Voltage (V)"].astype(int).astype(str) + " V"
    )

    order = sorted(df_plot["Cap–Vol group"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_plot, x="Cap–Vol group", y="Pred_Taper",
                order=order, palette="viridis", ax=ax, linewidth=1.2)
    sns.stripplot(data=df_plot, x="Cap–Vol group", y="Pred_Taper",
                  order=order, color="black", size=2.5, alpha=0.35, ax=ax)
    ax.set_xlabel("(Capacitance, Voltage) group", fontsize=11)
    ax.set_ylabel("Predicted Taper (deg)", fontsize=11)
    ax.set_title("Taper distribution across valid Cap–Vol groups", fontsize=13)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "taper_by_cap_vol_group.png")
    plt.savefig(path, dpi=600)
    plt.close()
    print(f"Saved taper-by-group plot to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Print-optimised Figure 21 supplements (unchanged logic, now constrained)
# ─────────────────────────────────────────────────────────────────────────────

def generate_print_optimized_supplements(df_all: pd.DataFrame):
    plot_df = df_all.copy()
    plot_df["Taper Bin"] = pd.cut(
        plot_df["Pred_Taper"],
        bins=[0, 1.5, 2.5, 3.5, 5],
        labels=["0-1.5°", "1.5-2.5°", "2.5-3.5°", ">3.5°"],
    )

    g = sns.FacetGrid(plot_df, col="Taper Bin", hue="Taper Bin",
                      palette="viridis", height=4)
    g.map(plt.scatter, "Pred_Entry", "Pred_Exit", s=20, alpha=0.6, edgecolor="none")
    g.set_axis_labels("Entry dia (µm)", "Exit dia (µm)")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Faceted Geometry Cloud (Split by Taper Intensity)")
    facet_path = os.path.join(PLOT_DIR, "Fig21_Supplement_Faceted.png")
    g.savefig(facet_path, dpi=600)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.kdeplot(data=plot_df, x="Pred_Entry", y="Pred_Exit",
                levels=5, color="black", linewidths=0.5, alpha=0.3)
    sc = plt.scatter(plot_df["Pred_Entry"], plot_df["Pred_Exit"],
                     c=plot_df["Pred_Taper"], cmap="viridis",
                     s=25, alpha=0.4, edgecolor="none")
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.title("Geometry Cloud with Density Contour Overlays")
    plt.colorbar(sc, label="Taper (deg)")
    contour_path = os.path.join(PLOT_DIR, "Fig21_Supplement_Contours.png")
    plt.savefig(contour_path, dpi=600)
    plt.close()
    print(f"Supplement plots: 1. {facet_path}\n2. {contour_path}")


def generate_manuscript_ready_plots(df_all: pd.DataFrame):
    plt.figure(figsize=(8, 8))
    g = sns.JointGrid(data=df_all, x="Pred_Entry", y="Pred_Exit", space=0)
    g.plot_joint(plt.scatter, c=df_all["Pred_Taper"], cmap="viridis",
                 s=15, alpha=0.3, edgecolor="none")
    g.plot_joint(sns.kdeplot, color="black", linewidths=1.2, levels=4, alpha=0.6)
    g.plot_marginals(sns.histplot, color="#2c3e50", fill=True, alpha=0.4, bins=30)
    g.set_axis_labels("Predicted Entry dia (µm)", "Predicted Exit dia (µm)")
    joint_path = os.path.join(PLOT_DIR, "Fig21_Joint_Distribution.png")
    g.savefig(joint_path, dpi=600)
    plt.close()

    df_plot = df_all.copy()
    median_taper = df_plot["Pred_Taper"].median()
    df_plot["Taper_Class"] = df_plot["Pred_Taper"].apply(
        lambda x: f"Precision (<{median_taper:.2f}°)"
        if x <= median_taper else f"High Taper (>{median_taper:.2f}°)"
    )
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_plot, x="Pred_Entry", y="Pred_Exit", hue="Taper_Class",
        palette={f"Precision (<{median_taper:.2f}°)":  "#2ecc71",
                 f"High Taper (>{median_taper:.2f}°)": "#e74c3c"},
        s=20, alpha=0.5, edgecolor="none"
    )
    plt.title("Functional Grouping: Precision vs. Taper-Dominant Design Space")
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.legend(title="Design Intent", loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    discrete_path = os.path.join(PLOT_DIR, "Fig21_Discrete_Classes.png")
    plt.savefig(discrete_path, dpi=600)
    plt.close()
    print(f"Manuscript plots: 1. {joint_path}\n2. {discrete_path}")


def generate_reviewer_requested_plots(df_all: pd.DataFrame):
    df_plot = df_all.copy()
    df_plot["Marker_Group"] = df_plot["Voltage (V)"].astype(str) + "V"

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot, x="Pred_Entry", y="Pred_Exit",
        hue="Pred_Taper", style="Marker_Group",
        palette="viridis", s=60, alpha=0.7, edgecolor="w", linewidth=0.5
    )
    plt.title("Figure 21 (Revised): Marker-Differentiated Geometry Cloud")
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.legend(title="Voltage Levels", bbox_to_anchor=(1.05, 1), loc="upper left")
    marker_path = os.path.join(PLOT_DIR, "Fig21_Marker_Differentiation.png")
    plt.savefig(marker_path, dpi=600, bbox_inches="tight")
    plt.close()

    g = sns.FacetGrid(df_plot, col="Marker_Group", hue="Pred_Taper",
                      palette="viridis", col_wrap=3, height=4)
    g.map(plt.scatter, "Pred_Entry", "Pred_Exit", s=30, alpha=0.6)
    g.set_axis_labels("Entry dia (µm)", "Exit dia (µm)")
    g.fig.suptitle("2D Projections of Geometry Clusters by Process Voltage", y=1.02)
    projection_path = os.path.join(PLOT_DIR, "Fig21_2D_Projections.png")
    g.savefig(projection_path, dpi=600)
    plt.close()

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(df_plot["Pred_Entry"], df_plot["Pred_Exit"],
                    C=df_plot["Pred_Taper"], gridsize=30, cmap="viridis",
                    reduce_C_function=np.mean, mincnt=1)
    plt.colorbar(hb, label="Mean Taper (deg)")
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Exit dia (µm)")
    plt.title("Advanced Geometry Cloud: Hexagonal Binning")
    hex_path = os.path.join(PLOT_DIR, "Fig21_Advanced_Hexbin.png")
    plt.savefig(hex_path, dpi=600)
    plt.close()
    print(f"Reviewer plots: 1. {marker_path}\n2. {projection_path}\n3. {hex_path}")


def generate_2x2_projection_grid(df_all: pd.DataFrame):
    unique_vols   = sorted(df_all["Voltage (V)"].unique())
    selected_vols = unique_vols[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    cmap = sns.color_palette("viridis", as_cmap=True)

    for i, vol in enumerate(selected_vols):
        subset = df_all[df_all["Voltage (V)"] == vol]
        ax = axes[i]
        sc = ax.scatter(
            subset["Pred_Entry"], subset["Pred_Exit"],
            c=subset["Pred_Taper"], cmap=cmap, s=25, alpha=0.4, edgecolor="none"
        )
        sns.kdeplot(data=subset, x="Pred_Entry", y="Pred_Exit",
                    ax=ax, color="black", linewidths=0.8, alpha=0.5, levels=4)
        ax.set_title(f"Projection: Voltage = {vol} V", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        if i >= 2: ax.set_xlabel("Entry dia (µm)")
        if i % 2 == 0: ax.set_ylabel("Exit dia (µm)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Predicted Taper (deg)")
    plt.subplots_adjust(right=0.9, hspace=0.2, wspace=0.1)
    grid_path = os.path.join(PLOT_DIR, "Fig21_2x2_Projection_Grid.png")
    plt.savefig(grid_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"2x2 projection grid saved to {grid_path}")


def generate_optimized_2x2_grid(df_all: pd.DataFrame):
    unique_vols   = sorted(df_all["Voltage (V)"].unique())
    selected_vols = unique_vols[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, vol in enumerate(selected_vols):
        subset = df_all[df_all["Voltage (V)"] == vol]
        ax = axes[i]
        sns.kdeplot(data=subset, x="Pred_Entry", y="Pred_Exit",
                    ax=ax, fill=True, alpha=0.1, color="gray", levels=5, thresh=0.1)
        sns.scatterplot(
            data=subset, x="Pred_Entry", y="Pred_Exit",
            hue="Pred_Taper", style="Capacitance (pF)",
            palette="viridis", s=70, alpha=0.8, ax=ax,
            edgecolor="black", linewidth=0.5
        )
        x_min, x_max = subset["Pred_Entry"].min(), subset["Pred_Entry"].max()
        y_min, y_max = subset["Pred_Exit"].min(), subset["Pred_Exit"].max()
        px = (x_max - x_min) * 0.15
        py = (y_max - y_min) * 0.15
        ax.set_xlim(x_min - px, x_max + px)
        ax.set_ylim(y_min - py, y_max + py)
        ax.set_title(f"Voltage: {vol} V", fontsize=11, fontweight="bold")
        ax.set_xlabel("Entry dia (µm)")
        ax.set_ylabel("Exit dia (µm)")
        ax.legend(title="Cap (pF)", loc="lower right", fontsize="x-small", frameon=True)

    norm = plt.Normalize(df_all["Pred_Taper"].min(), df_all["Pred_Taper"].max())
    sm   = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                 label="Predicted Taper (deg)")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    final_grid_path = os.path.join(PLOT_DIR, "Fig21_Final_Manuscript_Grid.png")
    plt.savefig(final_grid_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Optimized grid saved: {final_grid_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 0) Show and save constraint validity map
    plot_cap_vol_validity_map()

    # 1) Evaluate full (constrained) design space once
    df_all = evaluate_full_design_space()
    print_objective_definition()

    # 2) Show taper distribution by valid Cap–Vol group (new diagnostic plot)
    plot_taper_by_cap_vol_group(df_all)

    # 3) Global process scatter
    scatter_process_vs_taper(df_all)

    # 4) Inverse design for multiple targets
    targets = [
        (590.36, 588.94),
        (580.0,  540.0),
        (620.0,  580.0),
    ]

    all_inv_results = []
    for i, (target_entry, target_exit) in enumerate(targets):
        print(f"\n=== Inverse design: target Entry={target_entry}, Exit={target_exit} ===")
        inv_results = inverse_design(
            target_entry=target_entry,
            target_exit=target_exit,
            df_all=df_all,
            taper_max=None,
            taper_weight=1,
            top_k=5,
            verbose=(i == 0),
        )
        print(inv_results)
        # Sanity-check: all returned settings must be valid
        for _, row in inv_results.iterrows():
            assert is_valid_cap_vol(row["Capacitance (pF)"], row["Voltage (V)"]), \
                "BUG: invalid setting slipped through the inverse-design filter!"

        all_inv_results.append(inv_results)
        scatter_geometry_with_inverse_points(
            df_all, inv_results.head(1),
            tag=f"{int(target_entry)}_{int(target_exit)}"
        )
        scatter_geometry_cloud(df_all, tag="full_space")

        best_row  = inv_results.iloc[0]
        exp_match = lookup_experimental_match(best_row, df)
        if exp_match is not None:
            print("\nMatching experimental row:")
            print(exp_match[["Capacitance (pF)", "Voltage (V)", "TRS (rpm)",
                             "Feed rate (µm/s)", "Entry dia", "Exit dia"]])
        else:
            print("\nNo exact experimental match found for this setting.")

    # 5) Local sensitivity around best T2 setting
    base_row = all_inv_results[1].iloc[0]
    print("\n=== Local sensitivity around best inverse-designed setting (580, 540) ===")
    print(base_row)
    df_neighbors = local_sensitivity_cap_vol(base_row)
    print("\nLocal neighbourhood (sorted by Pred_Taper):")
    print(df_neighbors)
    plot_local_taper_heatmap(df_neighbors, base_row)

    # 6) Save inverse-design results
    for (te, tx), inv_df in zip(targets, all_inv_results):
        fname = f"inverse_results_target_{int(te)}_{int(tx)}.csv"
        inv_df.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
        print(f"Saved {fname}")

    # 7) Lambda sweep
    sweep_df   = lambda_sweep(580, 540, df_all)
    sweep_path = os.path.join(RESULTS_DIR, "lambda_sweep_590.36_588.94.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"Saved lambda sweep to {sweep_path}")

    # 8) Candidate targets
    den_min, den_max = float(df["Entry dia"].min()), float(df["Entry dia"].max())
    dex_min, dex_max = float(df["Exit dia"].min()), float(df["Exit dia"].max())
    candidate_targets = [
        {"Target_ID": "T1", "Target_Entry": 580.0, "Target_Exit": 540.0},
        {"Target_ID": "T2", "Target_Entry": 600.0, "Target_Exit": 560.0},
        {"Target_ID": "T3", "Target_Entry": 620.0, "Target_Exit": 580.0},
    ]
    for k in range(4, 11):
        alpha = (k - 4) / 6.0
        candidate_targets.append({
            "Target_ID":    f"T{k}",
            "Target_Entry": round(den_min + alpha * (den_max - den_min), 2),
            "Target_Exit":  round(dex_min + alpha * (dex_max - dex_min), 2),
        })
    ct_df = pd.DataFrame(candidate_targets)
    ct_df.to_csv(os.path.join(RESULTS_DIR, "candidate_targets_10.csv"), index=False)
    print(f"\n10 candidate targets:\n{ct_df}")

    # 9) Entry-only inverse design
    target_entry_single = 630
    inv_entry_only = inverse_design_entry_only(
        target_entry=target_entry_single,
        df_all=df_all,
        taper_max=None,
        taper_weight=1,
        top_k=5,
        verbose=True,
    )
    inv_entry_only_sorted = inv_entry_only.sort_values("ObjValue", ascending=True)
    entry_only_path = os.path.join(RESULTS_DIR,
                                   f"entry_only_inverse_{int(target_entry_single)}.csv")
    inv_entry_only_sorted.to_csv(entry_only_path, index=False)
    print(f"\nSaved entry-only inverse-design to {entry_only_path}")

    plt.figure(figsize=(6, 5))
    plt.scatter(
        inv_entry_only_sorted["Pred_Entry"], inv_entry_only_sorted["Pred_Taper"],
        c=inv_entry_only_sorted["Pred_Taper"], cmap="magma_r", s=60, edgecolor="black"
    )
    plt.xlabel("Predicted Entry dia (µm)")
    plt.ylabel("Predicted Taper (deg)")
    plt.title(f"Entry-only inverse design (target Den = {target_entry_single} µm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"entry_only_inverse_{int(target_entry_single)}.png")
    plt.savefig(plot_path, dpi=600)
    plt.close()
    print(f"Saved entry-only plot to {plot_path}")

    # 10) All supplementary and manuscript plots
    generate_print_optimized_supplements(df_all)
    generate_manuscript_ready_plots(df_all)
    generate_reviewer_requested_plots(df_all)
    generate_2x2_projection_grid(df_all)
    generate_optimized_2x2_grid(df_all)