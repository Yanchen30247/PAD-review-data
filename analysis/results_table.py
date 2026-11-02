# Results table for the pooled estimates reported in Section 4: k studies, m effect sizes, N participants,
# pooled estimate with 95% CI and p, heterogeneity (three-level I2, Cheung 2014), and a multilevel Egger test
# (standard error as a moderator in the same three-level model).
# Run: python results_table.py
import os
import numpy as np, pandas as pd
from run_rq_meta import load, OUT
from meta_core import ml_meta

D = load()
P = D[D.primary_ok]
P = P[~(P.study_id.eq("Steinhauser2018") & P.y_label.str.contains("tracking", case=False))]
B = P[P.y_stage.eq("S3") & P.zero]
BD = B[~B.no_dir]
O = P[P.x_stage.eq("S3") & P.y_stage.eq("S4")]
OC = O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]
OI = O[O.y_code.eq("INJURY")]
NVC = ["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]
VC = ["V_SCALE", "SPEED", "FOLLOW", "LANE", "OVERTAKE", "SIGNAL", "RACE"]
C = P[P.y_code.eq("CRASH_SELF") & P.level.isin(["driver", ""]) & P.zero & ~P.no_dir]


def row(rq, label, g, metric="z"):
    v = g.vz if metric == "z" else g.vlogor
    ok = g[metric].notna() & v.notna() & (v > 0)
    g = g[ok]; v = v[ok]
    y = g[metric].values; vv = v.values; cl = g.cluster.values
    r = ml_meta(y, vv, cl)
    tr = np.tanh if metric == "z" else np.exp
    w = 1 / vv; k_e = len(vv)
    v_typ = (k_e - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())
    tb, tw = r["tau2_between"], r["tau2_within"]
    i2 = (tb + tw) / (tb + tw + v_typ) * 100
    eg = ml_meta(y, vv, cl, np.column_stack([np.ones(len(y)), np.sqrt(vv)]))
    n = g.assign(nn=pd.to_numeric(g.n_eff, errors="coerce")).groupby("cluster").nn.max().sum()
    return dict(rq=rq, label=label, k=r["k"], m=r["m"], N=int(n) if n == n else None,
                est=tr(r["b"][0]), lo=tr(r["lo"][0]), hi=tr(r["hi"][0]), p=r["p"][0],
                tau2_between=tb, tau2_within=tw, I2=i2, egger_b=eg["b"][1], egger_p=eg["p"][1])


rows = [
    row("RQ1", "Triggering factors", BD[BD.x_stage.eq("S1")]),
    row("RQ1", "Younger age", B[B.x_code.eq("AGE")]),
    row("RQ1", "Male gender", B[B.x_code.eq("MALE")]),
    row("RQ1", "Crash or violation history", B[B.x_code.eq("HISTORY")]),
    row("RQ1", "More driving exposure", B[B.x_code.eq("EXPOSURE")]),
    row("RQ1", "Social norms approving risky driving", B[B.x_code.eq("SOCIAL_NORMS")]),
    row("RQ2", "Intentional processes", BD[BD.x_stage.eq("S2i")]),
    row("RQ2", "Anger", B[B.x_code.eq("ANGER")]),
    row("RQ2", "Attitudes favoring risk", B[B.x_code.eq("ATTITUDE")]),
    row("RQ2", "Sensation seeking", B[B.x_code.eq("SENSATION")]),
    row("RQ2", "Unintentional processes", BD[BD.x_stage.eq("S2u")]),
    row("RQ4", "Crash involvement, all behaviors (OR)", OC, "logor"),
    row("RQ4", "Crash involvement, violative (OR)", OC[OC.x_code.isin(VC)], "logor"),
    row("RQ4", "Crash involvement, non-violative (OR)", OC[OC.x_code.isin(NVC)], "logor"),
    row("RQ4", "Injury in crashes (OR)", OI, "logor"),
    row("RQ5", "PAD behaviors and crash involvement", C[C.x_stage.eq("S3")]),
    row("RQ5", "Intentional processes and crash involvement", C[C.x_stage.eq("S2i")]),
    row("RQ5", "Unintentional processes and crash involvement", C[C.x_stage.eq("S2u")]),
]
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "results_table.csv"), index=False)
pd.set_option("display.width", 250)
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
