# Table S3: every pooled estimate reported in the manuscript or its figures, plus the factor-level estimates that are not shown
# in a figure (Section 3.3.2). Same model and subsets as results_table.py; Egger test only with at least four studies.
# Writes results/results_table_s3.csv. Run: python results_table_s3.py
import json, os
import numpy as np, pandas as pd
from run_rq_meta import load, OUT
from meta_core import ml_meta

HERE = os.path.dirname(os.path.abspath(__file__))
D = load()
P = D[D.primary_ok]
P = P[~(P.study_id.eq("Steinhauser2018") & P.y_label.str.contains("tracking", case=False))]
B = P[P.y_stage.eq("S3") & P.zero]
BD = B[~B.no_dir]
O = P[P.x_stage.eq("S3") & P.y_stage.eq("S4")]
OC = O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]
OI = O[O.y_code.eq("INJURY")]
OS = O[O.y_code.eq("SEVERE")]
NVC = ["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]
VC = ["V_SCALE", "SPEED", "FOLLOW", "LANE", "OVERTAKE", "SIGNAL", "RACE"]
C = P[P.y_code.eq("CRASH_SELF") & P.level.isin(["driver", ""]) & P.zero & ~P.no_dir]
CF = P[P.y_code.eq("CRASH_SELF") & P.level.isin(["driver", ""]) & P.zero]


def stats(y, vv, cl):
    r = ml_meta(y, vv, cl)
    w = 1 / vv
    v_typ = (len(vv) - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())
    tb, tw = r["tau2_between"], r["tau2_within"]
    i2 = (tb + tw) / (tb + tw + v_typ) * 100
    eg = ml_meta(y, vv, cl, np.column_stack([np.ones(len(y)), np.sqrt(vv)]))["p"][1] if r["k"] >= 4 else np.nan
    return r, i2, eg


def row(rq, label, g, metric="z"):
    v = g.vz if metric == "z" else g.vlogor
    ok = g[metric].notna() & v.notna() & (v > 0)
    g = g[ok]
    v = v[ok]
    r, i2, eg = stats(g[metric].values, v.values, g.cluster.values)
    tr = np.tanh if metric == "z" else np.exp
    n = g.assign(nn=pd.to_numeric(g.n_eff, errors="coerce")).groupby("cluster").nn.max().sum()
    return dict(rq=rq, label=label, k=r["k"], m=r["m"], N=int(n), est=tr(r["b"][0]), lo=tr(r["lo"][0]), hi=tr(r["hi"][0]), I2=i2, egger_p=eg)


SUB = lambda code: B[B.x_code.eq(code)]
rows = [
    row("RQ1", "Triggering factors and PAD behaviors (r)", BD[BD.x_stage.eq("S1")]),
    row("RQ1", "Triggering factors and violative behaviors (r)", BD[BD.x_stage.eq("S1") & BD.y_code.eq("V")]),
    row("RQ1", "Triggering factors and non-violative behaviors (r)", BD[BD.x_stage.eq("S1") & BD.y_code.eq("NV")]),
    row("RQ1", "Younger age and PAD behaviors (r)", SUB("AGE")),
    row("RQ1", "Male gender and PAD behaviors (r)", SUB("MALE")),
    row("RQ1", "Crash or violation history and PAD behaviors (r)", SUB("HISTORY")),
    row("RQ1", "More driving exposure and PAD behaviors (r)", SUB("EXPOSURE")),
    row("RQ1", "Social norms approving risky driving and PAD behaviors (r)", SUB("SOCIAL_NORMS")),
    row("RQ1", "Less driving experience and PAD behaviors (r)", SUB("EXPERIENCE")),
    row("RQ1", "Alcohol or drug use and PAD behaviors (r)", SUB("ALCOHOL_DRUG")),
    row("RQ1", "Congestion and PAD behaviors (r)", SUB("CONGESTION")),
    row("RQ1", "Passengers and PAD behaviors (r)", SUB("PASSENGERS")),
    row("RQ1", "Higher education and PAD behaviors (r)*", SUB("EDUCATION")),
    row("RQ1", "Higher income and PAD behaviors (r)*", SUB("INCOME")),
    row("RQ1", "Employment and PAD behaviors (r)*", SUB("EMPLOYED")),
    row("RQ1", "Work or commercial driving and PAD behaviors (r)*", SUB("OCCUPATION")),
    row("RQ1", "Married or having children and PAD behaviors (r)*", SUB("FAMILY")),
    row("RQ1", "Rural roads and PAD behaviors (r)*", SUB("ROAD")),
    row("RQ1", "Vehicle characteristics and PAD behaviors (r)*", SUB("VEHICLE")),
    row("RQ1", "Older vehicle and PAD behaviors (r)*", SUB("VEH_AGE")),
    row("RQ1", "Higher vehicle status and PAD behaviors (r)*", SUB("VEH_STATUS")),
    row("RQ1", "Other background factors and PAD behaviors (r)*", SUB("BG_OTHER")),
    row("RQ1", "Other external factors and PAD behaviors (r)*", SUB("EXT_OTHER")),
    row("RQ2", "Intentional processes and PAD behaviors (r)", BD[BD.x_stage.eq("S2i")]),
    row("RQ2", "Intentional processes and violative behaviors (r)", BD[BD.x_stage.eq("S2i") & BD.y_code.eq("V")]),
    row("RQ2", "Intentional processes and non-violative behaviors (r)", BD[BD.x_stage.eq("S2i") & BD.y_code.eq("NV")]),
    row("RQ2", "Anger and PAD behaviors (r)", SUB("ANGER")),
    row("RQ2", "Normlessness and PAD behaviors (r)", B[B.x_sub.eq("NORMLESSNESS")]),
    row("RQ2", "Lower agreeableness and PAD behaviors (r)", SUB("PERS_A")),
    row("RQ2", "Attitudes favoring risk and PAD behaviors (r)", SUB("ATTITUDE")),
    row("RQ2", "Sensation seeking and PAD behaviors (r)", SUB("SENSATION")),
    row("RQ2", "Poorer emotion regulation and PAD behaviors (r)", SUB("EMOTION_REG")),
    row("RQ2", "Impulsivity and PAD behaviors (r)", SUB("IMPULSIVITY")),
    row("RQ2", "Higher neuroticism and PAD behaviors (r)", SUB("PERS_N")),
    row("RQ2", "Self-beliefs and PAD behaviors (r)", SUB("SELF_BELIEFS")),
    row("RQ2", "Intention to drive aggressively and PAD behaviors (r)", SUB("INTENTION")),
    row("RQ2", "Extraversion and PAD behaviors (r)*", SUB("PERS_E")),
    row("RQ2", "Other intentional processes and PAD behaviors (r)*", SUB("PSY_OTHER")),
    row("RQ2", "Unintentional processes and PAD behaviors (r)", BD[BD.x_stage.eq("S2u")]),
    row("RQ2", "Errors and lapses and PAD behaviors (r)", SUB("ERR_LAPSE")),
    row("RQ2", "Cognitive failures and PAD behaviors (r)", SUB("COG_FAILURE")),
    row("RQ2", "Inattention and PAD behaviors (r)", SUB("INATTENTION")),
]

# RQ3: co-occurrence (same data as Figure 7b) and item frequency (same data as Figure 7a)
vnv = json.load(open(os.path.join(HERE, "stage3", "stage3_vnv_results.json"), encoding="utf-8"))
co = pd.DataFrame(vnv["cooccurrence_rows"])
co = co[co.id.ne("Yousaf2024_CN")]
z, v, cl = np.arctanh(co.r.values), 1 / (co.n.values - 3), co.id.str.replace("_PK", "").values
r, i2, eg = stats(z, v, cl)
rows.append(dict(rq="RQ3", label="Violative and non-violative behaviors, within samples (r)", k=r["k"], m=r["m"], N=int(co.n.sum()),
                 est=np.tanh(r["b"][0]), lo=np.tanh(r["lo"][0]), hi=np.tanh(r["hi"][0]), I2=i2, egger_p=eg))
ra = json.load(open(os.path.join(HERE, "stage3", "stage3_rulea_results.json"), encoding="utf-8"))
inp = pd.DataFrame(ra["dbq_inputs"])
dbq = pd.DataFrame(ra["dbq"])
dbq = dbq[dbq.label.ne("Angry pursuit")].sort_values("mean", ascending=False)  # angry pursuit is not shown in Figure 7a
for it in dbq.itertuples():
    n = inp[inp.label.eq(it.label)].n.sum()
    rows.append(dict(rq="RQ3", label=f"{it.label}, pooled mean frequency (0\u20135)", k=it.studies, m=it.k, N=int(n), est=it.mean, lo=it.ci_low, hi=it.ci_high, I2=it.I2, egger_p=np.nan))

rows += [
    row("RQ4", "PAD behaviors and crash involvement (OR)", OC, "logor"),
    row("RQ4", "Violative behaviors and crash involvement (OR)", OC[OC.x_code.isin(VC)], "logor"),
    row("RQ4", "Non-violative behaviors and crash involvement (OR)", OC[OC.x_code.isin(NVC)], "logor"),
    row("RQ4", "PAD behaviors and injury in crashes (OR)", OI, "logor"),
    row("RQ4", "Aggressive or reckless driving code and injury in crashes (OR)", OI[OI.x_code.eq("AGG_CODE")], "logor"),
    row("RQ4", "PAD behaviors and severe or fatal injury (OR)", OS, "logor"),
    row("RQ5", "PAD behaviors and self-reported crash involvement (r)", C[C.x_stage.eq("S3")]),
    row("RQ5", "Intentional processes and self-reported crash involvement (r)", C[C.x_stage.eq("S2i")]),
    row("RQ5", "Anger and self-reported crash involvement (r)", CF[CF.x_code.eq("ANGER")]),
    row("RQ5", "Unintentional processes and self-reported crash involvement (r)", C[C.x_stage.eq("S2u")]),
    row("RQ5", "Errors and lapses and self-reported crash involvement (r)", CF[CF.x_code.eq("ERR_LAPSE")]),
    row("Sensitivity", "Unintentional processes and PAD behaviors, without same-questionnaire effect sizes (r)", BD[BD.x_stage.eq("S2u") & ~BD.same_instr]),
    row("Sensitivity", "Triggering factors and observed or recorded PAD behaviors (r)", BD[BD.x_stage.eq("S1") & BD.obj_y.eq(1)]),
    row("Sensitivity", "Intentional processes and observed or recorded PAD behaviors (r)", BD[BD.x_stage.eq("S2i") & BD.obj_y.eq(1)]),
    row("Sensitivity", "PAD behaviors and recorded crash involvement (OR)", OC[OC.y_code.eq("CRASH_REC")], "logor"),
]
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "results_table_s3.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 90)
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
