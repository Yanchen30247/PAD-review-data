# Study-level classification behind results/vote_counts.csv (Section 3.3.5): one call per study and factor.
# Run: python vote_by_study.py  -> results/vote_by_study.csv
import os
import pandas as pd
from run_rq_meta import load, OUT


def call(s):
    d = s.direction.astype(str).str.strip()
    pos, neg = d.eq("+").sum(), d.eq("-").sum()
    if pos + neg and pos / (pos + neg) >= 0.7:
        return "more"
    if pos + neg and neg / (pos + neg) >= 0.7:
        return "fewer"
    return "unclear"


D = load()
V = D[~D.primary.str.lower().eq("no") & ~D.sens_only & ~D.doubt]
rows = []
for st in ["S1", "S2i", "S2u"]:
    for code, g in V[V.x_stage.eq(st) & V.y_stage.eq("S3")].groupby("x_code"):
        for cl, s in g.groupby("cluster"):
            rows.append(dict(rq="RQ1" if st == "S1" else "RQ2", stage=st, factor=code, study_id=cl, call=call(s), results=len(s)))
NVC = ["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]
VC = ["V_SCALE", "SPEED", "FOLLOW", "LANE", "OVERTAKE", "SIGNAL", "RACE"]
O = V[V.x_stage.eq("S3") & V.y_stage.eq("S4")]
for lab, og in [("crash", O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]), ("injury", O[O.y_code.eq("INJURY")]), ("severe", O[O.y_code.eq("SEVERE")])]:
    for cls, codes in [("violative", VC), ("non-violative", NVC), ("mixed", ["MIXED_SCALE"]), ("aggressive code", ["AGG_CODE"])]:
        for cl, s in og[og.x_code.isin(codes)].groupby("cluster"):
            rows.append(dict(rq="RQ4", stage=lab, factor=cls, study_id=cl, call=call(s), results=len(s)))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "vote_by_study.csv"), index=False)
# the counts must reproduce results/vote_counts.csv
c = pd.read_csv(os.path.join(OUT, "vote_counts.csv"))
for r in c.itertuples():
    g = t[t.rq.eq(r.rq) & t.stage.eq(r.stage) & t.factor.eq(r.factor)]
    assert (g.call.eq("more").sum(), g.call.eq("unclear").sum(), g.call.eq("fewer").sum()) == (r.more, r.unclear, r.fewer), r
print("vote_by_study.csv matches vote_counts.csv:", len(c), "factors")
