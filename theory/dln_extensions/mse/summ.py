import pickle,glob,numpy as np,warnings; warnings.filterwarnings("ignore")
from mse_sim import LOSSES
import analyse
K=("m1","m2","m3","m4")
CF=["base","ffrac=0.1","ffrac=0.9","ffrac=1","noise_pt=0.1","noise_pt=0.5","fresh","fresh,noise_ft=0.5","S=0.1","S=0.3","S=3"]
out={}
for cn in CF:
    for dn in "CABD":
        S=[pickle.load(open(f,"rb")) for f in sorted(glob.glob(f"results/{cn}__{dn}__sim*.pkl"))]
        if len(S)<5: continue
        pre={k:np.nanmean([s["pre_end"][k] for s in S]) for k in K}; U={k:np.nanmean([s["ref_untrain_end"][k] for s in S]) for k in K}; L={k:np.nanmean([s["ref_unlearn_end"][k] for s in S]) for k in K}
        d0u=np.mean([s["d_pre"]["untrain"] for s in S]); d0l=np.mean([s["d_pre"]["unlearn"] for s in S]); out[(cn,dn)]=dict(pre=pre,U=U,L=L,d0u=d0u,d0l=d0l,loss={})
        for li,l in enumerate(LOSSES):
            e={k:np.nanmean([s["ft"][k][-1,li] for s in S]) for k in K}; du=np.nanmean([s["ft"]["d_untrain"][:,li] for s in S],0); dl=np.nanmean([s["ft"]["d_unlearn"][:,li] for s in S],0)
            out[(cn,dn)]["loss"][l]=dict(end=e,du_min=np.nanmin(du),du_t=np.nanargmin(du)*0.1,du_end=du[-1],dl_min=np.nanmin(dl),dl_t=np.nanargmin(dl)*0.1,dl_end=dl[-1])
pickle.dump(out,open("summary.pkl","wb"))
for (cn,dn),o in out.items():
    z=o["loss"]["forget to zero"]; e=z["end"]; print(f"{cn:18s} {dn} pre m1 {o['pre']['m1']:.3f} m4 {o['pre']['m4']:.3f} | f2z end m1 {e['m1']:.3f} m2 {e['m2']:.3f} m3 {e['m3']:.3f} m4 {e['m4']:.3f} | U: m2 {o['U']['m2']:.3f} m4 {o['U']['m4']:.3f} L: m3 {o['L']['m3']:.3f} m4 {o['L']['m4']:.3f} | d_untr {o['d0u']:.3f}->{z['du_min']:.3f}@{z['du_t']:.1f} end {z['du_end']:.3f} | d_unl {o['d0l']:.3f}->{z['dl_min']:.3f}@{z['dl_t']:.1f} end {z['dl_end']:.3f}")
rows=analyse.agreement(CF); pickle.dump(rows,open("agreement.pkl","wb"))
good=[r for r in rows if r[2] in ("forget to zero","negated target -0.5","negated target -1") and not (r[4]==0 and r[3] in ("m2","m3"))]
def st(R):
    z=np.array([r[8] for r in R]); rel=np.array([abs(r[6]-r[7])/max(abs(r[7]),0.02) for r in R]); return f"n={len(R)} within2 {100*np.mean(abs(z)<2):.0f}% within3 {100*np.mean(abs(z)<3):.0f}% median gap {100*np.median(rel):.1f}% p90 {100*np.percentile(rel,90):.1f}%"
print("ALL three losses:",st(good))
for cn in CF: print(" ",cn,st([r for r in good if r[0]==cn]))
nz=[r for r in rows if r[2]=="forget to noise" and not (r[4]==0 and r[3] in ("m2","m3"))]; print("noise:",st(nz))
