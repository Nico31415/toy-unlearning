import numpy as np
import pandas as pd

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

def isclose_any(series, values, atol=1e-12, rtol=0.0):
    vals = list(values)
    if len(vals) == 0:
        return np.zeros(len(series), dtype=bool)
    s = pd.to_numeric(series, errors="coerce")
    mask = np.zeros(len(s), dtype=bool)
    for v in vals:
        mask |= np.isclose(s.to_numpy(), float(v), atol=atol, rtol=rtol, equal_nan=False)
    return mask

# Panel A setup
EXPERIMENT = "exp1"
RHO_PT = 0.1
RHO_FT = 0.1
omega_order = [0.0, 0.5, 1.0]

rep = rep_all_df.copy()
emp = emp_all_df.copy()
rep = ensure_numeric(rep, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])
emp = ensure_numeric(emp, ["alpha","omega","rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit"])

rep = rep[rep["experiment"] == EXPERIMENT].copy()
emp = emp[emp["experiment"] == EXPERIMENT].copy()

print(f"After exp filter: rep={len(rep)}, emp={len(emp)}")

rep = rep[np.isclose(rep["rho_pt"], RHO_PT)]
emp = emp[np.isclose(emp["rho_pt"], RHO_PT)]
print(f"After rho_pt filter: rep={len(rep)}, emp={len(emp)}")

rep = rep[np.isclose(rep["rho_ft"], RHO_FT)]
emp = emp[np.isclose(emp["rho_ft"], RHO_FT)]
print(f"After rho_ft filter: rep={len(rep)}, emp={len(emp)}")

rep = rep[np.isclose(rep["lambda_pt"], 0.0)]
emp = emp[np.isclose(emp["lambda_pt"], 0.0)]
print(f"After lambda filter: rep={len(rep)}, emp={len(emp)}")

rep = rep[np.isclose(rep["gamma_reinit"], 0.0)]
emp = emp[np.isclose(emp["gamma_reinit"], 0.0)]
print(f"After gamma filter: rep={len(rep)}, emp={len(emp)}")

rep = rep[isclose_any(rep["omega"], omega_order, atol=1e-12, rtol=0.0)]
emp = emp[isclose_any(emp["omega"], omega_order, atol=1e-12, rtol=0.0)]
print(f"After omega filter: rep={len(rep)}, emp={len(emp)}")
print(f"Rep omega unique: {sorted(rep['omega'].unique())}")
print(f"Emp omega unique: {sorted(emp['omega'].unique())}")

base_cols = [c for c in ["rho_pt","rho_ft","c_pt","lambda_pt","gamma_reinit","ft_teacher_norm"]
             if c in rep.columns and c in emp.columns]
print(f"Base cols: {base_cols}")

rep["base_key"] = make_combo_key(rep, base_cols)
emp["base_key"] = make_combo_key(emp, base_cols)

print(f"\nRep base_keys: {rep['base_key'].unique()[:10]}")
print(f"Emp base_keys: {emp['base_key'].unique()[:10]}")
print(f"Rep unique base_keys: {len(rep['base_key'].unique())}")
print(f"Emp unique base_keys: {len(emp['base_key'].unique())}")

# Check for omega=0.5 specifically
rep_05 = rep[np.isclose(rep["omega"], 0.5, atol=1e-12, rtol=0.0)]
print(f"\nRep rows with omega~=0.5: {len(rep_05)}")
if len(rep_05) > 0:
    print(f"  Sample omega values: {rep_05['omega'].head(10).tolist()}")
    print(f"  Sample c_pt values: {rep_05['c_pt'].unique()}")
