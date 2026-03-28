import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator


rep_all_df = pd.read_csv("rep_all_df.csv")
emp_all_df = pd.read_csv("emp_all_df.csv")

rep = rep_all_df.copy()
emp = emp_all_df.copy()

TOL = 1e-12

def ensure_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_combo_key(df, cols, float_round=12):
    d = df.copy()
    parts = []
    for c in cols:
        if c not in d.columns:
            continue
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce").round(float_round)
            parts.append(s.map(lambda x: "nan" if pd.isna(x) else f"{x:.{float_round}g}"))
        else:
            parts.append(s.astype(str))
    if not parts:
        return pd.Series([""] * len(d), index=d.index)
    out = parts[0].astype(str)
    for p in parts[1:]:
        out = out + "|" + p.astype(str)
    return out

def format_params_label(row, cols, float_round=6):
    bits = []
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        if isinstance(v, (float, np.floating, int, np.integer)):
            bits.append(f"{c}={float(v):.{float_round}g}")
        else:
            bits.append(f"{c}={v}")
    return ", ".join(bits)



rep_metric = "mse_best"
emp_metric = "final_param_mse"

# -----------------------
# Sanity
# -----------------------
for c in ["alpha", "omega"]:
    if c not in rep_all_df.columns or c not in emp_all_df.columns:
        raise KeyError(f"Both dfs must contain '{c}'")

if rep_metric not in rep_all_df.columns:
    raise KeyError(f"Replica df missing '{rep_metric}'")
if emp_metric not in emp.columns or "seed" not in emp.columns:
    raise KeyError("Empirical df missing metric or 'seed'")

rep = ensure_numeric(rep, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])
emp = ensure_numeric(emp, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])

# -----------------------
# Define BASE param combo (EXCLUDES omega)
# -----------------------
candidate_base_cols = ["rho_pt", "rho_ft", "c_pt", "lambda_pt", "gamma_reinit", "ft_teacher_norm"]
base_cols = [c for c in candidate_base_cols if (c in rep.columns and c in emp.columns)]
if not base_cols:
    raise ValueError("No shared base columns found")

rep["base_key"] = make_combo_key(rep, base_cols)
emp["base_key"] = make_combo_key(emp, base_cols)

# Full key (used to filter replica to empirical support)
full_cols = ["omega"] + base_cols
rep["full_key"] = make_combo_key(rep, full_cols)
emp["full_key"] = make_combo_key(emp, full_cols)

# -----------------------
# Empirical stats: mean ± 95% CI
# -----------------------
emp_stats = (
    emp.dropna(subset=["alpha","omega","base_key",emp_metric])
       .groupby(["omega","base_key","alpha"], as_index=False)
       .agg(
           mean=(emp_metric,"mean"),
           std=(emp_metric,"std"),
           n=("seed","nunique"),
       )
)
emp_stats["sem"] = emp_stats["std"] / np.sqrt(emp_stats["n"].clip(lower=1))
emp_stats["ci95"] = 1.96 * emp_stats["sem"]

# valid base combos
base_keys = sorted(emp_stats["base_key"].unique())
if not base_keys:
    raise ValueError("No empirical base_key combinations found")

# empirical-supported full keys per omega
emp_fullkeys_by_omega = {
    om: set(emp.loc[np.isclose(emp["omega"], om), "full_key"].unique())
    for om in [0.0, 0.5, 1.0]
}

# labels for color legend
emp_base_rows = emp.groupby("base_key", as_index=False).first().set_index("base_key")
rep_base_rows = rep.groupby("base_key", as_index=False).first().set_index("base_key")

def get_base_label(bk):
    row = emp_base_rows.loc[bk] if bk in emp_base_rows.index else rep_base_rows.loc[bk]
    return format_params_label(row, base_cols)


# -----------------------
# Global style (set ONCE)
# -----------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "legend.fontsize": 15,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 1.2,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "errorbar.capsize": 4,

    "image.cmap": "viridis",
})

# -----------------------
# Helpers (shared)
# -----------------------
def ensure_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_combo_key(df, cols, float_round=12):
    d = df.copy()
    parts = []
    for c in cols:
        if c not in d.columns:
            continue
        s = d[c]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce").round(float_round)
            parts.append(s.map(lambda x: "nan" if pd.isna(x) else f"{x:.{float_round}g}"))
        else:
            parts.append(s.astype(str))
    if not parts:
        return pd.Series([""] * len(d), index=d.index)
    out = parts[0].astype(str)
    for p in parts[1:]:
        out = out + "|" + p.astype(str)
    return out

def viridis_map(values, lo=0.15, hi=0.95, cmap_name="viridis"):
    vals = list(values)
    cmap = plt.get_cmap(cmap_name)
    return {
        v: cmap(lo + (hi - lo) * i / max(1, len(vals) - 1))
        for i, v in enumerate(vals)
    }

def add_top_legend(ax, handles, labels, title, ncol=None, y=1.22):
    if ncol is None:
        ncol = len(labels)
    return ax.legend(
        handles, labels, title=title, frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        columnspacing=1.2,
        handletextpad=0.6,
    )

def format_axes(ax, ylim, yticks):
    ax.set_xlim(0.0, 0.5)
    ax.set_xticks([0.0, 0.25, 0.5])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Expected MSE")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

def emp_ci95_stats(emp, group_cols, metric):
    s = (emp.groupby(group_cols, as_index=False)
           .agg(mean=(metric, "mean"), std=(metric, "std"), n=("seed", "nunique")))
    s["sem"] = s["std"] / np.sqrt(s["n"].clip(lower=1))
    s["ci95"] = 1.96 * s["sem"]
    return s

def isclose_any(series, values, atol=1e-12, rtol=1e-9):
    """Vectorized 'series is close to any value in values' for float columns."""
    vals = list(values)
    if len(vals) == 0:
        return np.zeros(len(series), dtype=bool)
    s = pd.to_numeric(series, errors="coerce")
    mask = np.zeros(len(s), dtype=bool)
    for v in vals:
        mask |= np.isclose(s.to_numpy(), float(v), atol=atol, rtol=rtol, equal_nan=False)
    return mask

# -----------------------
# Load + numeric once (shared by all panels)
# Assumes rep_all_df and emp_all_df exist in your environment.
# -----------------------
rep_metric = "mse_best"
emp_metric = "final_param_mse"

num_cols_rep = ["alpha", rep_metric, "omega", "rho_pt", "rho_ft", "c_pt", "lambda_pt", "gamma_reinit"]
num_cols_emp = ["alpha", emp_metric, "omega", "rho_pt", "rho_ft", "c_pt", "lambda_pt", "gamma_reinit", "seed"]

rep_all = ensure_numeric(rep_all_df.copy(), num_cols_rep)
emp_all = ensure_numeric(emp_all_df.copy(), num_cols_emp)

# -----------------------
# Panel A: vary c_PT (exp1; rho_pt=rho_ft=0.1; lambda=0; gamma=0; omega in {0,0.5,1})
# -----------------------
def panel_cpt(ax, rep_all, emp_all):
    EXPERIMENT = "exp1"
    RHO_PT = 0.1
    RHO_FT = 0.1
    omega_order = [0.0, 0.5, 1.0]
    linestyle_map = {0.0: ":", 0.5: "--", 1.0: "-"}

    rep = rep_all.copy()
    emp = emp_all.copy()
    rep = rep[rep["experiment"] == EXPERIMENT].copy()
    emp = emp[emp["experiment"] == EXPERIMENT].copy()

    for df in (rep, emp):
        df.dropna(subset=["alpha", "omega"], inplace=True)

    rep = rep[np.isclose(rep["rho_pt"], RHO_PT)]
    emp = emp[np.isclose(emp["rho_pt"], RHO_PT)]
    rep = rep[np.isclose(rep["rho_ft"], RHO_FT)]
    emp = emp[np.isclose(emp["rho_ft"], RHO_FT)]
    rep = rep[np.isclose(rep["lambda_pt"], 0.0)]
    emp = emp[np.isclose(emp["lambda_pt"], 0.0)]
    rep = rep[np.isclose(rep["gamma_reinit"], 0.0)]
    emp = emp[np.isclose(emp["gamma_reinit"], 0.0)]
    # robust float matching for omega
    rep = rep[isclose_any(rep["omega"], omega_order, atol=1e-12, rtol=0.0)]
    emp = emp[isclose_any(emp["omega"], omega_order, atol=1e-12, rtol=0.0)]

    base_cols = [c for c in ["rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit","ft_teacher_norm"]
                 if c in rep.columns and c in emp.columns]

    rep["base_key"] = make_combo_key(rep, base_cols)
    emp["base_key"] = make_combo_key(emp, base_cols)

    emp_stats = emp_ci95_stats(emp, ["omega","base_key","alpha"], emp_metric)

    tmp = emp.groupby("base_key", as_index=False).first()
    base_to_cpt = dict(zip(tmp["base_key"], tmp["c_pt"]))
    base_keys = sorted(emp_stats["base_key"].unique())
    ordered_base_keys = sorted(base_keys, key=lambda k: base_to_cpt[k])

    color_map = viridis_map(ordered_base_keys)

    # replica curves
    for om in omega_order:
        rep_o = rep[rep["omega"] == om]
        for bk, d in rep_o.groupby("base_key"):
            if bk not in color_map:
                continue
            d = d.sort_values("alpha")
            ax.plot(d["alpha"], d[rep_metric], color=color_map[bk],
                    linestyle=linestyle_map[om], linewidth=2.2, zorder=2)

    # empirical points
    for om in omega_order:
        s = emp_stats[emp_stats["omega"] == om]
        for bk, d in s.groupby("base_key"):
            if bk not in color_map:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"], d["mean"], yerr=d["ci95"],
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[bk], zorder=3)

    format_axes(ax, ylim=(-0.05, 3.0), yticks=[0.0, 1.5, 3.0])

    # top legend: c_PT
    handles = [Line2D([0],[0], color=color_map[bk], lw=2.5) for bk in ordered_base_keys]
    labels = [f"{base_to_cpt[bk]:.3g}" for bk in ordered_base_keys]
    add_top_legend(ax, handles, labels, title=r"$c_{PT}$", ncol=len(labels), y=1.24)

# -----------------------
# Panel B: vary gamma_reinit (exp1; rho_pt=rho_ft=0.1; c_pt=1e-3; lambda=0; omega in {0,0.5,1})
# -----------------------
def panel_gamma(ax, rep_all, emp_all):
    EXPERIMENT = "exp1"
    RHO_PT = 0.1
    RHO_FT = 0.1
    C_PT = 1e-3
    LAMBDA = 0.0
    OMEGA_VALUES = [0.0, 0.5, 1.0]
    ls_map = {0.0: ":", 0.5: "--", 1.0: "-"}

    rep = rep_all.copy()
    emp = emp_all.copy()
    rep = rep[rep["experiment"] == EXPERIMENT].copy()
    emp = emp[emp["experiment"] == EXPERIMENT].copy()

    def filt(df):
        df = df[np.isclose(df["rho_pt"], RHO_PT, atol=1e-9)].copy()
        df = df[np.isclose(df["rho_ft"], RHO_FT, atol=1e-9)].copy()
        df = df[np.isclose(df["c_pt"], C_PT, atol=1e-12)].copy()
        df = df[np.isclose(df["lambda_pt"], LAMBDA, atol=1e-12)].copy()
        df = df[isclose_any(df["omega"], OMEGA_VALUES, atol=1e-12, rtol=0.0)].copy()
        df = df.dropna(subset=["alpha","omega","gamma_reinit"]).copy()
        return df

    rep = filt(rep)
    emp = filt(emp)

    gamma_vals = sorted(set(rep["gamma_reinit"].dropna().unique()) |
                        set(emp["gamma_reinit"].dropna().unique()))
    color_map = viridis_map(gamma_vals)

    rep["full_key"] = make_combo_key(rep, ["omega","gamma_reinit"])
    emp["full_key"] = make_combo_key(emp, ["omega","gamma_reinit"])
    rep = rep[rep["full_key"].isin(emp["full_key"].dropna().unique())].copy()

    emp_stats = emp_ci95_stats(emp, ["omega","gamma_reinit","alpha"], emp_metric)

    for g in gamma_vals:
        for om in OMEGA_VALUES:
            d = rep[np.isclose(rep["gamma_reinit"], g, atol=1e-12, rtol=1e-9) &
                    np.isclose(rep["omega"], om, atol=1e-12, rtol=0.0)]
            if d.empty: 
                continue
            d = d.sort_values("alpha")
            ax.plot(d["alpha"], d[rep_metric], color=color_map[g],
                    linestyle=ls_map[om], linewidth=2.2, zorder=2)

    for g in gamma_vals:
        for om in OMEGA_VALUES:
            d = emp_stats[np.isclose(emp_stats["gamma_reinit"], g, atol=1e-12, rtol=1e-9) &
                          np.isclose(emp_stats["omega"], om, atol=1e-12, rtol=0.0)]
            if d.empty:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"], d["mean"], yerr=d["ci95"],
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[g], zorder=3)

    format_axes(ax, ylim=(-0.05, 2.0), yticks=[0.0, 1.0, 2.0])

    handles = [Line2D([0],[0], color=color_map[g], lw=2.5) for g in gamma_vals]
    labels  = [f"{g:g}" for g in gamma_vals]
    add_top_legend(ax, handles, labels, title=r"$\gamma_{\mathrm{reinit}}$", ncol=len(labels), y=1.24)

# -----------------------
# Panel C: vary lambda_pt (rho_pt=rho_ft=0.1; c_pt=1e-3; gamma=0; omega in {0,0.5,1})
# top legend lambda_pt; right legend omega
# -----------------------
def panel_lambda_omega(ax, rep_all, emp_all):
    RHO_PT = 0.1
    RHO_FT = 0.1
    C_PT   = 1e-3
    GAMMA  = 0.0
    OMEGA_VALUES = [0.0, 0.5, 1.0]
    ls_map = {0.0: ":", 0.5: "--", 1.0: "-"}
    LAMBDA_VALUES = [-1e-3, -0.99e-3, 0.0, 0.99e-3]
    EXCLUDE_LAMBDA_EQ_MINUS_C = True

    rep = rep_all.copy()
    emp = emp_all.copy()

    def filt(df):
        df = df[np.isclose(df["rho_pt"], RHO_PT, atol=1e-9)].copy()
        df = df[np.isclose(df["rho_ft"], RHO_FT, atol=1e-9)].copy()
        df = df[np.isclose(df["c_pt"], C_PT, atol=1e-12)].copy()
        df = df[np.isclose(df["gamma_reinit"], GAMMA, atol=1e-12)].copy()
        df = df[isclose_any(df["omega"], OMEGA_VALUES, atol=1e-12, rtol=0.0)].copy()
        df = df[isclose_any(df["lambda_pt"], LAMBDA_VALUES, atol=1e-12, rtol=1e-9)].copy()
        df = df.dropna(subset=["alpha","omega","lambda_pt"]).copy()
        return df

    rep = filt(rep)
    emp = filt(emp)

    if EXCLUDE_LAMBDA_EQ_MINUS_C:
        rep = rep[~np.isclose(rep["lambda_pt"], -rep["c_pt"], atol=1e-12)].copy()
        emp = emp[~np.isclose(emp["lambda_pt"], -emp["c_pt"], atol=1e-12)].copy()

    lambda_vals = sorted(set(rep["lambda_pt"].dropna().unique()) | set(emp["lambda_pt"].dropna().unique()))
    omega_present = sorted(set(rep["omega"].dropna().unique()) | set(emp["omega"].dropna().unique()))
    omega_present = [om for om in OMEGA_VALUES if om in omega_present]

    color_map = viridis_map(lambda_vals)

    rep["full_key"] = make_combo_key(rep, ["omega","lambda_pt"])
    emp["full_key"] = make_combo_key(emp, ["omega","lambda_pt"])
    rep = rep[rep["full_key"].isin(emp["full_key"].dropna().unique())].copy()

    emp_stats = emp_ci95_stats(emp, ["omega","lambda_pt","alpha"], emp_metric)

    for lam in lambda_vals:
        for om in omega_present:
            d = rep[np.isclose(rep["lambda_pt"], lam, atol=1e-12) & np.isclose(rep["omega"], om, atol=1e-12)]
            if d.empty:
                continue
            d = d.dropna(subset=["alpha", rep_metric]).sort_values("alpha")
            ax.plot(d["alpha"].to_numpy(), d[rep_metric].to_numpy(),
                    color=color_map[lam], linestyle=ls_map.get(om, "-"),
                    linewidth=2.2, zorder=2)

    for lam in lambda_vals:
        for om in omega_present:
            d = emp_stats[np.isclose(emp_stats["lambda_pt"], lam, atol=1e-12) & np.isclose(emp_stats["omega"], om, atol=1e-12)]
            if d.empty:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"].to_numpy(), d["mean"].to_numpy(),
                        yerr=d["ci95"].to_numpy(),
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[lam], zorder=3)

    format_axes(ax, ylim=(-0.05, 2.0), yticks=[0.0, 1.0, 2.0])

    # top legend: lambda
    lam_handles = [Line2D([0],[0], color=color_map[lam], lw=2.5) for lam in lambda_vals]
    lam_labels  = [f"{lam:.8g}" for lam in lambda_vals]
    add_top_legend(ax, lam_handles, lam_labels, title=r"$\lambda_{PT}$",
                   ncol=min(len(lam_labels), 4), y=1.24)

    # right legend: omega
    omega_handles = [Line2D([0],[0], color="black", lw=2, linestyle=ls_map.get(om, "-")) for om in omega_present]
    omega_labels  = [f"{om:g}" for om in omega_present]
    ax.legend(
        omega_handles, omega_labels, title=r"$\omega$", frameon=False,
        loc="upper left", bbox_to_anchor=(1.02, 0.70), bbox_transform=ax.transAxes
    )

# -----------------------
# Panel D: (your exp2 / omega=0 / vary rho_ft) — no legends in screenshot
# -----------------------
def panel_rhoft_cpt(ax, rep_all, emp_all):
    EXPERIMENT = "exp2"
    OMEGA = 0.0
    RHO_PT = 0.1
    RHO_FT_VALUES = [0.01, 0.1, 0.9]
    ls_map = {0.01: ":", 0.1: "--", 0.9: "-"}  # rho_ft style

    rep = rep_all.copy()
    emp = emp_all.copy()
    if "experiment" in rep.columns:
        rep = rep[rep["experiment"] == EXPERIMENT].copy()
    if "experiment" in emp.columns:
        emp = emp[emp["experiment"] == EXPERIMENT].copy()

    def filt(df):
        df = df[np.isclose(df["omega"], OMEGA, atol=1e-9)].copy()
        df = df[np.isclose(df["rho_pt"], RHO_PT, atol=1e-9)].copy()
        df = df[isclose_any(df["rho_ft"], RHO_FT_VALUES, atol=1e-12, rtol=1e-9)].copy()
        df = df[np.isclose(df["lambda_pt"], 0.0, atol=1e-9)].copy()
        df = df[np.isclose(df["gamma_reinit"], 0.0, atol=1e-9)].copy()
        df = df.dropna(subset=["alpha","rho_ft"]).copy()
        return df

    rep = filt(rep)
    emp = filt(emp)

    # color by "base combo" excluding rho_ft (c_pt drives ordering)
    candidate_base_cols = ["omega","rho_pt","c_pt","lambda_pt","gamma_reinit","ft_teacher_norm"]
    base_cols = [c for c in candidate_base_cols if (c in rep.columns and c in emp.columns)]
    rep["base_key"] = make_combo_key(rep, base_cols)
    emp["base_key"] = make_combo_key(emp, base_cols)

    full_cols = base_cols + ["rho_ft"]
    rep["full_key"] = make_combo_key(rep, full_cols)
    emp["full_key"] = make_combo_key(emp, full_cols)

    rep = rep[rep["full_key"].isin(set(emp["full_key"].dropna().unique()))].copy()

    emp_stats = emp_ci95_stats(emp, ["rho_ft","base_key","full_key","alpha"], emp_metric)

    tmp = emp.dropna(subset=["base_key","c_pt"]).groupby("base_key", as_index=False).first()
    base_to_cpt = dict(zip(tmp["base_key"], tmp["c_pt"]))
    base_keys = sorted(emp_stats["base_key"].unique())
    ordered_base_keys = sorted(base_keys, key=lambda bk: float(base_to_cpt.get(bk, np.inf)))
    color_map = viridis_map(ordered_base_keys)

    for rf in RHO_FT_VALUES:
        d_rf = rep[np.isclose(rep["rho_ft"], rf, atol=1e-12, rtol=1e-9)]
        for bk, d in d_rf.groupby("base_key"):
            if bk not in color_map:
                continue
            d = d.dropna(subset=["alpha", rep_metric]).sort_values("alpha")
            ax.plot(d["alpha"].to_numpy(), d[rep_metric].to_numpy(),
                    color=color_map[bk], linestyle=ls_map[rf], linewidth=2.2, zorder=2)

    for rf in RHO_FT_VALUES:
        s = emp_stats[np.isclose(emp_stats["rho_ft"], rf, atol=1e-12, rtol=1e-9)]
        for bk, d in s.groupby("base_key"):
            if bk not in color_map:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"].to_numpy(), d["mean"].to_numpy(), yerr=d["ci95"].to_numpy(),
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[bk], zorder=3)

    format_axes(ax, ylim=(-0.05, 3.0), yticks=[0.0, 1.5, 3.0])

# -----------------------
# Panel E: exp2 / omega=0 / vary rho_ft and gamma (no legends in screenshot)
# -----------------------
def panel_gamma_rhoft(ax, rep_all, emp_all):
    EXPERIMENT = "exp2"
    OMEGA = 0.0
    RHO_PT = 0.1
    RHO_FT_VALUES = [0.01, 0.1, 0.9]
    C_PT = 1e-3
    LAMBDA = 0.0
    ls_map = {0.01: ":", 0.1: "--", 0.9: "-"}

    rep = rep_all.copy()
    emp = emp_all.copy()
    if "experiment" in rep.columns:
        rep = rep[rep["experiment"] == EXPERIMENT].copy()
    if "experiment" in emp.columns:
        emp = emp[emp["experiment"] == EXPERIMENT].copy()

    def filt(df):
        df = df[np.isclose(df["omega"], OMEGA, atol=1e-9)].copy()
        df = df[np.isclose(df["rho_pt"], RHO_PT, atol=1e-9)].copy()
        df = df[isclose_any(df["rho_ft"], RHO_FT_VALUES, atol=1e-12, rtol=1e-9)].copy()
        df = df[np.isclose(df["c_pt"], C_PT, atol=1e-12)].copy()
        df = df[np.isclose(df["lambda_pt"], LAMBDA, atol=1e-12)].copy()
        df = df.dropna(subset=["alpha","rho_ft","gamma_reinit"]).copy()
        return df

    rep = filt(rep)
    emp = filt(emp)

    gamma_vals = sorted(set(rep["gamma_reinit"].dropna().unique()) |
                        set(emp["gamma_reinit"].dropna().unique()))
    color_map = viridis_map(gamma_vals)

    rep["full_key"] = make_combo_key(rep, ["rho_ft","gamma_reinit"])
    emp["full_key"] = make_combo_key(emp, ["rho_ft","gamma_reinit"])
    rep = rep[rep["full_key"].isin(emp["full_key"].dropna().unique())].copy()

    emp_stats = emp_ci95_stats(emp, ["rho_ft","gamma_reinit","alpha"], emp_metric)

    for g in gamma_vals:
        for rf in RHO_FT_VALUES:
            d = rep[np.isclose(rep["gamma_reinit"], g, atol=1e-12, rtol=1e-9) &
                    np.isclose(rep["rho_ft"], rf, atol=1e-12, rtol=1e-9)]
            if d.empty:
                continue
            d = d.dropna(subset=["alpha", rep_metric]).sort_values("alpha")
            ax.plot(d["alpha"].to_numpy(), d[rep_metric].to_numpy(),
                    color=color_map[g], linestyle=ls_map[rf], linewidth=2.2, zorder=2)

    for g in gamma_vals:
        for rf in RHO_FT_VALUES:
            d = emp_stats[np.isclose(emp_stats["gamma_reinit"], g, atol=1e-12, rtol=1e-9) &
                          np.isclose(emp_stats["rho_ft"], rf, atol=1e-12, rtol=1e-9)]
            if d.empty:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"].to_numpy(), d["mean"].to_numpy(), yerr=d["ci95"].to_numpy(),
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[g], zorder=3)

    format_axes(ax, ylim=(-0.05, 2.0), yticks=[0.0, 1.0, 2.0])

# -----------------------
# Panel F: omega=1 / vary rho_ft; color by lambda; right legend rho_pt (actually rho_ft in your code)
# -----------------------
def panel_lambda_rhoft(ax, rep_all, emp_all):
    OMEGA = 1.0
    RHO_PT = 0.1
    C_PT = 1e-3
    GAMMA = 0.0
    RHO_FT_KEEP = [0.01, 0.04, 0.1]
    ls_map = {0.01: ":", 0.04: "--", 0.1: "-"}

    rep = rep_all.copy()
    emp = emp_all.copy()

    def filt(df):
        df = df[np.isclose(df["omega"], OMEGA, atol=1e-9)].copy()
        df = df[np.isclose(df["rho_pt"], RHO_PT, atol=1e-9)].copy()
        df = df[np.isclose(df["c_pt"], C_PT, atol=1e-12)].copy()
        df = df[np.isclose(df["gamma_reinit"], GAMMA, atol=1e-12)].copy()
        df = df[isclose_any(df["rho_ft"], RHO_FT_KEEP, atol=1e-12, rtol=1e-9)].copy()
        df = df.dropna(subset=["alpha","rho_ft","lambda_pt"]).copy()
        return df

    rep = filt(rep)
    emp = filt(emp)

    # Exclude lambda == -c
    rep = rep[~np.isclose(rep["lambda_pt"], -rep["c_pt"], atol=1e-12)].copy()
    emp = emp[~np.isclose(emp["lambda_pt"], -emp["c_pt"], atol=1e-12)].copy()

    rho_ft_present = sorted(set(rep["rho_ft"].dropna().unique()) | set(emp["rho_ft"].dropna().unique()))
    lambda_vals = sorted(set(rep["lambda_pt"].dropna().unique()) | set(emp["lambda_pt"].dropna().unique()))
    color_map = viridis_map(lambda_vals)

    rep["full_key"] = make_combo_key(rep, ["rho_ft","lambda_pt"])
    emp["full_key"] = make_combo_key(emp, ["rho_ft","lambda_pt"])
    rep = rep[rep["full_key"].isin(emp["full_key"].dropna().unique())].copy()

    emp_stats = emp_ci95_stats(emp, ["rho_ft","lambda_pt","alpha"], emp_metric)

    for lam in lambda_vals:
        for rf in rho_ft_present:
            d = rep[np.isclose(rep["lambda_pt"], lam, atol=1e-12) & np.isclose(rep["rho_ft"], rf, atol=1e-12)]
            if d.empty:
                continue
            d = d.dropna(subset=["alpha", rep_metric]).sort_values("alpha")
            ax.plot(d["alpha"].to_numpy(), d[rep_metric].to_numpy(),
                    color=color_map[lam], linestyle=ls_map.get(rf, "-"),
                    linewidth=2.2, zorder=2)

    for lam in lambda_vals:
        for rf in rho_ft_present:
            d = emp_stats[np.isclose(emp_stats["lambda_pt"], lam, atol=1e-12) & np.isclose(emp_stats["rho_ft"], rf, atol=1e-12)]
            if d.empty:
                continue
            d = d.sort_values("alpha")
            ax.errorbar(d["alpha"].to_numpy(), d["mean"].to_numpy(), yerr=d["ci95"].to_numpy(),
                        fmt="o", ms=5, capsize=4, elinewidth=1.6,
                        color=color_map[lam], zorder=3)

    format_axes(ax, ylim=(-0.05, 1.25), yticks=[0.0, 0.625, 1.25])

    # right legend: rho_ft (your screenshot label might say rho_PT; adjust title if needed)
    rf_handles = [Line2D([0],[0], color="black", lw=2, linestyle=ls_map.get(rf, "-")) for rf in rho_ft_present]
    rf_labels  = [f"{rf:g}" for rf in rho_ft_present]
    ax.legend(
        rf_handles, rf_labels, title=r"$\rho_{FT}$", frameon=False,
        loc="upper left", bbox_to_anchor=(1.02, 0.70), bbox_transform=ax.transAxes
    )

# -----------------------
# Build the 2x3 figure
# -----------------------
fig, axes = plt.subplots(2, 3, figsize=(17.5, 7.5))

# spacing tuned so top-row per-axes legends fit + right-side legends fit
# fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.88, wspace=0.35, hspace=0.35)
fig.subplots_adjust(left=0.06, right=0.88, bottom=0.10, top=0.92, wspace=0.35, hspace=0.35)

panel_cpt(axes[0, 0], rep_all, emp_all)
panel_gamma(axes[0, 1], rep_all, emp_all)
panel_lambda_omega(axes[0, 2], rep_all, emp_all)

panel_rhoft_cpt(axes[1, 0], rep_all, emp_all)
panel_gamma_rhoft(axes[1, 1], rep_all, emp_all)
panel_lambda_rhoft(axes[1, 2], rep_all, emp_all)

plt.savefig("figure3_detailed.png", dpi=300)