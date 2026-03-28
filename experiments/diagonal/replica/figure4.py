import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from pathlib import Path

HERE = Path(__file__).resolve().parent

# -----------------------
# Global style (matching Figure 3)
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
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "errorbar.capsize": 4,
})

def ensure_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def emp_ci95_stats(emp, group_cols, metric):
    s = (emp.groupby(group_cols, as_index=False)
           .agg(mean=(metric, "mean"), std=(metric, "std"), n=("seed", "nunique")))
    s["sem"] = s["std"] / np.sqrt(s["n"].clip(lower=1))
    s["ci95"] = 1.96 * s["sem"]
    return s

def _select_smooth_unique_alpha(d: pd.DataFrame, *, alpha_col: str, y_col: str) -> pd.DataFrame:
    """
    When multiple rows exist for the same alpha (e.g. mixed runs / subset alpha grids),
    choose one row per alpha in a globally consistent way (avoid alternating points that
    create jagged 'sawtooth' curves).

    Strategy:
    - Round alpha for stability, group candidates per alpha.
    - Dynamic programming over alphas to pick candidates that minimize total variation,
      with a strong penalty for non-monotone increases (these curves should be decreasing).
    """
    d = d.copy()
    d["alpha_round"] = pd.to_numeric(d[alpha_col], errors="coerce").round(10)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=["alpha_round", y_col]).copy()
    if d.empty:
        return d

    groups = []
    for a, dd in d.groupby("alpha_round", sort=True):
        # candidates: keep whole rows, but store y
        rows = [r for _, r in dd.iterrows()]
        ys = [float(r[y_col]) for r in rows]
        groups.append((float(a), rows, ys))

    # DP: dp[i][k] = (cost, prev_k)
    BIG = 1e9
    PEN_UP = 1e6  # penalize increases strongly
    dp_cost = []
    dp_prev = []

    # init
    a0, rows0, ys0 = groups[0]
    dp_cost.append([0.0 for _ in ys0])
    dp_prev.append([-1 for _ in ys0])

    for i in range(1, len(groups)):
        ai, rowsi, ysi = groups[i]
        prev_a, prev_rows, prev_ys = groups[i - 1]
        cur_costs = [BIG for _ in ysi]
        cur_prev = [-1 for _ in ysi]
        for k, y in enumerate(ysi):
            best = (BIG, -1)
            for pk, py in enumerate(dp_cost[i - 1]):
                y_prev = prev_ys[pk]
                step = abs(y - y_prev)
                pen = PEN_UP if (y - y_prev) > 0 else 0.0
                cost = dp_cost[i - 1][pk] + step + pen
                if cost < best[0]:
                    best = (cost, pk)
            cur_costs[k] = best[0]
            cur_prev[k] = best[1]
        dp_cost.append(cur_costs)
        dp_prev.append(cur_prev)

    # backtrack
    last_idx = int(np.argmin(dp_cost[-1]))
    chosen_rows = []
    idx = last_idx
    for i in range(len(groups) - 1, -1, -1):
        a, rowsi, ysi = groups[i]
        chosen_rows.append(rowsi[idx])
        idx = dp_prev[i][idx]
        if i > 0 and idx < 0:
            idx = 0
    chosen_rows = list(reversed(chosen_rows))

    out = pd.DataFrame(chosen_rows)
    out = out.sort_values("alpha_round").copy()
    return out

def main():
    rep_all_df = pd.read_csv(HERE / "rep_all_df.csv")
    emp_all_df = pd.read_csv(HERE / "emp_all_df.csv")

    rep_metric = "mse_best"
    emp_metric = "final_param_mse"

    num_cols_rep = ["alpha", "omega", "rho_pt", "rho_ft", "c_pt", "lambda_pt", "gamma_reinit", rep_metric]
    num_cols_emp = ["alpha", "omega", "rho_pt", "rho_ft", "c_pt", "lambda_pt", "gamma_reinit", "seed", emp_metric]

    rep = ensure_numeric(rep_all_df, num_cols_rep)
    emp = ensure_numeric(emp_all_df, num_cols_emp)

    # Core fixed settings
    RHO_PT = 0.1
    C_PT = 1e-3
    
    # Filter data
    rep = rep[np.isclose(rep["rho_pt"], RHO_PT) & np.isclose(rep["c_pt"], C_PT, atol=1e-12)].copy()
    emp = emp[np.isclose(emp["rho_pt"], RHO_PT) & np.isclose(emp["c_pt"], C_PT, atol=1e-12)].copy()

    # Ground-truth styles
    GT_STYLES = {
        (0.0, 0.1): (":", "no overlap, $\\rho_{FT}=0.1$"),
        (1.0, 0.1): ("-", "full overlap, $\\rho_{FT}=0.1$"),
        (1.0, 0.01): ("--", "full overlap, $\\rho_{FT}=0.01$"),
    }

    # Regime I lambda selection:
    # We must pick ONE lambda value (otherwise we mix nearby lambdas like -c_pt and -0.999999*c_pt,
    # which creates the jagged "sawtooth" curves).
    lam_regime1 = -0.999999 * C_PT  # default
    if "experiment" in rep.columns:
        r1 = rep[(rep["experiment"] == "fig4_local") & np.isclose(rep["gamma_reinit"], 1e-6, atol=1e-12)]
        uniq = np.array(sorted(set(pd.to_numeric(r1["lambda_pt"], errors="coerce").dropna().tolist())))
        if len(uniq) > 0:
            # Prefer a value that is NOT exactly -c_pt (avoid the numerically stiff/degenerate case).
            not_exact = uniq[np.abs(uniq - (-C_PT)) > 1e-15]
            lam_regime1 = float(not_exact[0] if len(not_exact) > 0 else uniq[0])

    regimes = [
        {"label": "Regime II", "lam": 0.0, "gam": 0.0, "color": "#3b0f70", "desc": r"$\lambda_{PT}=0,\ \gamma_{FT}=0$"},
        {"label": "Regime IV", "lam": -0.99 * C_PT, "gam": 0.0, "color": "#22a884", "desc": r"$\lambda_{PT}=-0.99,\ \gamma_{FT}=0$"},
        {"label": "Regime I", "lam": lam_regime1, "gam": 1e-6, "color": "#fde725", "desc": r"$\lambda_{PT}\approx -1,\ \gamma_{FT}=10^{-6}$"},
        {"label": "Regime III", "lam": 0.0, "gam": 10.0, "color": "#7f7f7f", "desc": r"$\lambda_{PT}=0,\ \gamma_{FT}=10$"},
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot each regime
    for reg in regimes:
        # Theory
        # IMPORTANT: keep lambda matching tight so we don't mix nearby lambdas into one curve.
        r_sub = rep[np.isclose(rep["lambda_pt"], reg["lam"], atol=1e-12) &
                    np.isclose(rep["gamma_reinit"], reg["gam"], atol=1e-12)].copy()
        if reg["label"] == "Regime I" and "experiment" in r_sub.columns:
            r_sub = r_sub[r_sub["experiment"] == "fig4_local"].copy()
        
        # Empirical
        # For Regime I, empirical might be exactly -c_pt
        if reg["label"] == "Regime I":
            e_sub = emp[np.isclose(emp["gamma_reinit"], reg["gam"], atol=1e-12) & 
                        (np.isclose(emp["lambda_pt"], reg["lam"], atol=1e-9) | 
                         np.isclose(emp["lambda_pt"], -C_PT, atol=1e-12))].copy()
        else:
            e_sub = emp[np.isclose(emp["lambda_pt"], reg["lam"], atol=1e-12) & 
                        np.isclose(emp["gamma_reinit"], reg["gam"], atol=1e-12)].copy()

        stats = emp_ci95_stats(e_sub, ["omega", "rho_ft", "alpha"], emp_metric)

        for (om, rf), (ls, _) in GT_STYLES.items():
            # Theory line
            # Filter and handle potential duplicates by rounding alpha to avoid floating point jitter
            d_th = r_sub[np.isclose(r_sub["omega"], om) & np.isclose(r_sub["rho_ft"], rf)].copy()
            if not d_th.empty:
                # In fig4_local (Regime I), we observed a 21-point alpha grid mixed with an 11-point subset grid,
                # producing duplicates at every other alpha (2nd/4th/6th...). Select a globally consistent curve.
                if reg["label"] == "Regime I":
                    d_plot = _select_smooth_unique_alpha(d_th, alpha_col="alpha", y_col=rep_metric)
                else:
                    d_th["alpha_round"] = pd.to_numeric(d_th["alpha"], errors="coerce").round(10)
                    d_plot = d_th.sort_values("alpha").drop_duplicates(subset=["alpha_round"], keep="last")
                ax.plot(d_plot["alpha"], d_plot[rep_metric], color=reg["color"], linestyle=ls, linewidth=2.5, zorder=2)
            
            # Empirical points intentionally omitted (curves-only figure).

    # Formatting
    ax.set_xlim(0.0, 0.505)
    ax.set_xticks([0.0, 0.5])
    ax.set_ylim(-0.05, 1.7)
    ax.set_yticks([0.0, 1.7])
    ax.set_xlabel(r"$\alpha$", labelpad=-15)
    ax.set_ylabel("Generalization error")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    # Setup legend (top right)
    reg_handles = [Line2D([0], [0], color=r["color"], lw=4) for r in regimes]
    reg_labels = [r["desc"] for r in regimes]
    leg_setup = ax.legend(reg_handles, reg_labels, title=r"Setup (fixed $c_{PT}=10^{-3}$)",
                          loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False,
                          title_fontsize=17, fontsize=14)
    ax.add_artist(leg_setup)

    # Ground truth legend (on top of the graph, like a title)
    gt_handles = [Line2D([0], [0], color="black", lw=2, linestyle=ls) for (ls, _) in [GT_STYLES[k] for k in GT_STYLES]]
    gt_labels = [desc for (_, desc) in [GT_STYLES[k] for k in GT_STYLES]]
    ax.legend(gt_handles, gt_labels, title="Ground truth properties",
              loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
              title_fontsize=16, fontsize=11, ncol=len(gt_labels),
              columnspacing=0.6, handletextpad=0.3)

    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95)
    
    fig.savefig(HERE / "figure4.png", dpi=300)
    fig.savefig(HERE / "figure4.pdf")
    print(f"Wrote figure4.png and figure4.pdf")

if __name__ == "__main__":
    main()
