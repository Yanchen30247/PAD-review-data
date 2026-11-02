# Direction-of-effect synthesis (vote counting) from the same extraction as the meta-analyses, including results that
# could not be converted to an effect size. One call per study (cluster) and factor: a direction when at least 70% of the
# results with a known sign agree, otherwise no clear difference; a study whose results have no known sign also counts as
# no clear difference. Two-sided sign test when at least four studies have a direction (Methods 3.3.5).
# Run: python vote_counts.py  -> results/vote_counts.csv
import os
import pandas as pd
from math import comb
from run_rq_meta import load, OUT


def sign_p(a, b):
    n = a + b
    if n < 4:
        return None
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def calls(g):
    out = []
    for cl, s in g.groupby("cluster"):
        d = s.direction.astype(str).str.strip()
        pos, neg = d.eq("+").sum(), d.eq("-").sum()
        if pos + neg and pos / (pos + neg) >= 0.7:
            out.append("more")
        elif pos + neg and neg / (pos + neg) >= 0.7:
            out.append("fewer")
        else:
            out.append("unclear")
    return pd.Series(out).value_counts().reindex(["more", "unclear", "fewer"], fill_value=0)


D = load()
V = D[~D.primary.str.lower().eq("no") & ~D.sens_only & ~D.doubt]
rows = []
# RQ1 and RQ2: factors and processes with PAD behaviors (zero-order and adjusted results)
for st in ["S1", "S2i", "S2u"]:
    g0 = V[V.x_stage.eq(st) & V.y_stage.eq("S3")]
    for code, g in g0.groupby("x_code"):
        c = calls(g)
        rows.append(dict(rq="RQ1" if st == "S1" else "RQ2", stage=st, factor=code, studies=g.cluster.nunique(),
                         more=c["more"], unclear=c["unclear"], fewer=c["fewer"], sign_p=sign_p(c["more"], c["fewer"]),
                         convertible_studies=g[g.r.notna()].cluster.nunique()))
# RQ4: behaviors with crash involvement (driver level) and injury in crashes
NVC = ["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]
VC = ["V_SCALE", "SPEED", "FOLLOW", "LANE", "OVERTAKE", "SIGNAL", "RACE"]
O = V[V.x_stage.eq("S3") & V.y_stage.eq("S4")]
for out_lab, og in [("crash", O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]),
                    ("injury", O[O.y_code.eq("INJURY")]), ("severe", O[O.y_code.eq("SEVERE")])]:
    for cls, codes in [("violative", VC), ("non-violative", NVC), ("mixed", ["MIXED_SCALE"]), ("aggressive code", ["AGG_CODE"])]:
        g = og[og.x_code.isin(codes)]
        if g.empty:
            continue
        c = calls(g)
        rows.append(dict(rq="RQ4", stage=out_lab, factor=cls, studies=g.cluster.nunique(), more=c["more"],
                         unclear=c["unclear"], fewer=c["fewer"], sign_p=sign_p(c["more"], c["fewer"]),
                         convertible_studies=g[g.r.notna()].cluster.nunique()))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "vote_counts.csv"), index=False)
pd.set_option("display.width", 200)
print(t.to_string(index=False))
