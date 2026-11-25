"""Tables: (1) theory against simulation, (2) end-of-run values of the four metrics against the two references."""
import numpy as np, pickle, glob, sys
from mse_sim import LOSSES
from run import CONFIGS, DATA
KEYS = ("m1", "m2", "m3", "m4")

def load(cn, dn):
    S = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"results/{cn}__{dn}__sim*.pkl"))]; T = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"results/{cn}__{dn}__th*.pkl"))]; return S, T

def agreement(sel=None, out=sys.stdout):
    rows = []
    for cn in (sel or CONFIGS):
        for dn in DATA:
            S, T = load(cn, dn)
            if len(S) < 2 or len(T) < 2: continue
            rec = S[0]["cfg"]["rec"]; dt = T[0]["dt"]; nT = len(T[0]["ft"][LOSSES[0]]["m1"]) - 1
            for li, l in enumerate(LOSSES):
                for key in KEYS + ("d_untrain", "d_unlearn"):
                    for frac in (0.0, 0.25, 0.5, 1.0):
                        it = int(round(frac * nT)); t = it * dt; isim = int(round(t / rec)); a = np.array([x["ft"][l][key][it] for x in T]); b = np.array([s["ft"][key][isim, li] for s in S])
                        if np.any(np.isnan(a)) or np.any(np.isnan(b)): continue
                        se = np.hypot(a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))); rows.append((cn, dn, l, key, frac, t, a.mean(), b.mean(), (a.mean() - b.mean()) / max(se, 1e-12), se))
    return rows

def summary(rows):
    z = np.array([r[8] for r in rows]); rel = np.array([abs(r[6] - r[7]) / max(abs(r[7]), 0.02) for r in rows]); print(f"{len(rows)} comparisons: {np.mean(np.abs(z) < 2) * 100:.0f}% within 2 s.e., {np.mean(np.abs(z) < 3) * 100:.0f}% within 3; median |theory - sim| / sim = {np.median(rel) * 100:.1f}%, 90th percentile {np.percentile(rel, 90) * 100:.1f}%; mean z = {z.mean():+.2f}")
    by = {}
    for r in rows: by.setdefault(r[0], []).append(r)
    for cn, rs in by.items():
        z = np.array([r[8] for r in rs]); rel = np.array([abs(r[6] - r[7]) / max(abs(r[7]), 0.02) for r in rs]); w = max(rs, key=lambda r: abs(r[6] - r[7]) / max(abs(r[7]), 0.02))
        print(f"  {cn:22s} n={len(rs):4d}  within 2 s.e. {np.mean(np.abs(z) < 2) * 100:3.0f}%  median rel. gap {np.median(rel) * 100:5.1f}%  mean z {z.mean():+.2f}   worst: data {w[1]}, {w[2]}, {w[3]}, t={w[5]:.1f}: {w[6]:.4f} vs {w[7]:.4f}")

def ends(cn, dn, tfrac=1.0):
    S, _ = load(cn, dn)
    if not S: return None
    i = int(round(tfrac * (len(S[0]["ft"]["m1"]) - 1))); out = {}
    for key in KEYS + ("d_untrain", "d_unlearn"):
        v = np.array([s["ft"][key][i] for s in S]); out[key] = (np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) / np.sqrt(len(S)))
    for nm in ("pre", "ref_untrain", "ref_unlearn"): out[nm] = {key: np.nanmean([s[nm + "_end"][key] for s in S]) for key in KEYS}
    return out

if __name__ == "__main__":
    sel = sys.argv[1].split(";") if len(sys.argv) > 1 else None; rows = agreement(sel); summary(rows); pickle.dump(rows, open("agreement.pkl", "wb"))
    for cn in (sel or CONFIGS):
        for dn in DATA:
            e = ends(cn, dn)
            if e is None: continue
            print(f"\n{cn} | data {dn} | end of finetuning (simulation).  pretrained: " + " ".join(f"{k} {e['pre'][k]:.3f}" for k in KEYS) + " | ideal untraining: " + " ".join(f"{k} {e['ref_untrain'][k]:.3f}" for k in KEYS) + " | ideal unlearning: " + " ".join(f"{k} {e['ref_unlearn'][k]:.3f}" for k in KEYS))
            for li, l in enumerate(LOSSES): print(f"   {l:20s} " + " ".join(f"{k} {e[k][0][li]:.3f}" for k in KEYS + ("d_untrain", "d_unlearn")))
