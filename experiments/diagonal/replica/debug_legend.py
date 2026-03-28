import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rep_all_df = pd.read_csv("rep_all_df.csv")
emp_all_df = pd.read_csv("emp_all_df.csv")

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

def isclose_any(series, values, atol=1e-12, rtol=0.0):
    vals = list(values)
    if len(vals) == 0:
        return np.zeros(len(series), dtype=bool)
    s = pd.to_numeric(series, errors="coerce")
    mask = np.zeros(len(s), dtype=bool)
    for v in vals:
        mask |= np.isclose(s.to_numpy(), float(v), atol=atol, rtol=rtol, equal_nan=False)
    return mask

def emp_ci95_stats(emp, group_cols, metric):
    s = (emp.groupby(group_cols, as_index=False)
           .agg(mean=(metric, "mean"), std=(metric, "std"), n=("seed", "nunique")))
    s["sem"] = s["std"] / np.sqrt(s["n"].clip(lower=1))
    s["ci95"] = 1.96 * s["sem"]
    return s

# Panel A setup
EXPERIMENT = "exp1"
RHO_PT = 0.1
RHO_FT = 0.1
omega_order = [0.0, 0.5, 1.0]
emp_metric = "final_param_mse"

rep = rep_all_df.copy()
emp = emp_all_df.copy()
rep = ensure_numeric(rep, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])
emp = ensure_numeric(emp, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])

rep = rep[rep["experiment"] == EXPERIMENT].copy()
emp = emp[emp["experiment"] == EXPERIMENT].copy()

rep = rep[np.isclose(rep["rho_pt"], RHO_PT)]
emp = emp[np.isclose(emp["rho_pt"], RHO_PT)]
rep = rep[np.isclose(rep["rho_ft"], RHO_FT)]
emp = emp[np.isclose(emp["rho_ft"], RHO_FT)]
rep = rep[np.isclose(rep["lambda_pt"], 0.0)]
emp = emp[np.isclose(emp["lambda_pt"], 0.0)]
rep = rep[np.isclose(rep["gamma_reinit"], 0.0)]
emp = emp[np.isclose(emp["gamma_reinit"], 0.0)]
rep = rep[isclose_any(rep["omega"], omega_order, atol=1e-12, rtol=0.0)]
emp = emp[isclose_any(emp["omega"], omega_order, atol=1e-12, rtol=0.0)]

base_cols = [c for c in ["rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"]
             if c in rep.columns and c in emp.columns]

rep["base_key"] = make_combo_key(rep, base_cols)
emp["base_key"] = make_combo_key(emp, base_cols)

emp_stats = emp_ci95_stats(emp, ["omega","base_key","alpha"], emp_metric)

tmp = emp.groupby("base_key", as_index=False).first()
base_to_cpt = dict(zip(tmp["base_key"], tmp["c_pt"]))

all_base_keys = sorted(set(rep["base_key"].unique()) | set(emp_stats["base_key"].unique()))
ordered_base_keys = sorted(all_base_keys, key=lambda k: base_to_cpt.get(k, 0.0))

color_map = viridis_map(ordered_base_keys)

print(f"ordered_base_keys: {ordered_base_keys}")
print(f"base_to_cpt: {base_to_cpt}")
print(f"color_map keys: {list(color_map.keys())}")
print(f"\nLegend labels would be:")
for bk in ordered_base_keys:
    print(f"  {bk} -> c_PT={base_to_cpt.get(bk, 'MISSING'):.3g}")
