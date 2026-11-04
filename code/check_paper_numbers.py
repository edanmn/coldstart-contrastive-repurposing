"""Prints each new number the paper states next to its value recomputed from results/ CSVs."""
import pandas as pd, numpy as np
R = "results"
v2 = {t: pd.read_csv(f"{R}/membership_v2_{t}_summary.csv").set_index("cls") for t in
      ["ecfp4_p2", "ecfp4_p1", "chemberta_p2", "gcn_p2", "finlayson_p2", "random_p1"]}
v1r = pd.read_csv(f"{R}/membership_matched_random_control_summary.csv").set_index("cls")
e = v2["ecfp4_p2"]
def show(label, stated, val): print(f"{label:60s} stated {stated:>18s} | data {val}")
show("ECFP4 P2 mem matched range", "0.800-0.817", f"{e.mem_matched.min():.3f}-{e.mem_matched.max():.3f}")
show("random P2 mem matched range", "0.956-0.966", f"{v1r.membership_auroc_matched.min():.3f}-{v1r.membership_auroc_matched.max():.3f}")
fm = pd.concat([e.mem_full, v1r.membership_auroc_full])
show("full-encoder mem range (ECFP4+random)", "0.48-0.52", f"{fm.min():.3f}-{fm.max():.3f}")
g = v1r[v1r.holdout_minus_full > 0.1]
show("random hold gains (4 classes)", "+0.177..+0.359", f"{sorted(g.holdout_minus_full.round(3))}")
show("random matched for those", "-0.003..+0.110", f"{sorted(g.matched_minus_full.round(3))}")
show("random matched gain p min (Bonf .0083)", ">0.0083", f"{v1r[v1r.matched_minus_full>0].p_matched_vs_full.min():.4f}")
d = (e.sizehold - e.matched).round(3); show("size - matched SSRI/NSAID/AP", "0.074-0.136", f"{d[['SSRI','NSAID','Antipsychotic']].tolist()} p={e.p_sizehold_vs_matched[['SSRI','NSAID','Antipsychotic']].round(4).tolist()}")
show("AED size vs matched", "0.774/0.772 p=0.85", f"{e.sizehold['AED']:.3f}/{e.matched['AED']:.3f} p={e.p_sizehold_vs_matched['AED']:.3f}")
c = v2["chemberta_p2"]; show("ChemBERTa mem range", "0.79-0.82", f"{c.mem_matched.min():.3f}-{c.mem_matched.max():.3f}")
show("ChemBERTa SSRI matched gain", "+0.228 p=0.002", f"{c.matched_minus_full['SSRI']:+.3f} p={c.p_matched_vs_full['SSRI']:.4f}")
gm = pd.concat([v2['gcn_p2'].mem_matched, v2['finlayson_p2'].mem_matched]); show("GCN+D-MPNN mem range", "0.47-0.52", f"{gm.min():.3f}-{gm.max():.3f}")
p1 = v2["ecfp4_p1"]; show("P1 ECFP4 mem range", "0.60-0.66", f"{p1.mem_matched.min():.3f}-{p1.mem_matched.max():.3f}")
show("P1 ECFP4 full", "0.77-0.99 in five", f"{sorted(p1.full.round(3))}")
show("P1 BB matched gain", "+0.209", f"{p1.matched_minus_full['BetaBlocker']:+.3f} p={p1.p_matched_vs_full['BetaBlocker']:.3f}")
show("P1 sig negative classes", "Statin,SSRI,NSAID", f"{p1[(p1.matched_minus_full<0)&(p1.p_matched_vs_full<0.05)].index.tolist()}")
rp = v2["random_p1"]; show("random P1 mem / full", "0.78-0.81 / 0.66-0.95", f"{rp.mem_matched.min():.3f}-{rp.mem_matched.max():.3f} / {rp.full.min():.3f}-{rp.full.max():.3f}")
show("random P1 hold losses", "five classes", f"{int(((rp.hold_minus_full<0)&(rp.p_hold_vs_full<0.05)).sum())}")
ev = pd.read_csv(f"{R}/encoders_vs_ecfp4rf.csv"); f = ev[(ev.dataset=='p2')&(ev.arm=='full')]
for inp in ["gcn","finlayson"]: show(f"{inp} full vs RF min p", ">=0.15", f"{f[f.input==inp].p.min():.3f}")
for inp in ["ecfp4","chemberta"]:
    x=f[f.input==inp]; show(f"{inp} full loses to RF", "5 of 6, p<=0.010", f"{int(((x['diff']<0)&(x.p<0.05)).sum())} max p {x[(x['diff']<0)&(x.p<0.05)].p.max():.3f}")
show("AED ECFP4+RF on M splits", "0.777", f"{f[(f.input=='ecfp4')&(f.cls=='AED')].ecfp4_rf.iloc[0]:.3f}")
for t in ["ecfp4_p2","chemberta_p2","gcn_p2","finlayson_p2","ecfp4_p1"]:
    sp = pd.read_csv(f"{R}/similarity_preservation_{t}.csv").groupby("cls").mean(numeric_only=True)
    show(f"similarity {t} seen/unseen", "(see text)", f"{sp.rho_M_seen_full.min():.3f}-{sp.rho_M_seen_full.max():.3f} / {sp.rho_M_unseen_matched.min():.3f}-{sp.rho_M_unseen_matched.max():.3f}")
lo = pd.read_csv(f"{R}/lopo_matched_ecfp4_p2.csv"); show("drug-level ECFP4 P2 n / delta", "48 / +0.136", f"{len(lo)} / {(lo.matched-lo.full).mean():+.3f}")
