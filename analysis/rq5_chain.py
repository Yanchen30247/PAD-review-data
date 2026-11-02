# RQ5: stage-level pooled correlation matrix (three-level models on all constructs of each stage, risk-aligned)
# and one mediation model per earlier stage (X -> PAD-related behaviors -> crash involvement, with direct path).
# Intervals: parametric bootstrap (5000 draws) from the pooled Fisher z and their SEs.
# Run: python rq5_chain.py
import os
import numpy as np, pandas as pd
from run_rq_meta import load, fit, OUT

STAGES = ["S1", "S2i", "S2u", "S3", "S4"]
idx = {s: i for i, s in enumerate(STAGES)}


def stage_pairs(d, crash_codes):
    p = d[d.primary_ok & d.zero & ~d.no_dir]
    p = p[~(p.study_id.eq("Steinhauser2018") & p.y_label.str.contains("tracking", case=False))]
    beh = p[p.y_stage.eq("S3") & p.x_stage.isin(["S1", "S2i", "S2u"])]
    out = {f"{s}-S3": beh[beh.x_stage.eq(s)] for s in ["S1", "S2i", "S2u"]}
    crash = p[p.y_code.isin(crash_codes) & p.level.isin(["driver", ""])]
    out["S3-S4"] = crash[crash.x_stage.eq("S3")]
    for s in ["S1", "S2i", "S2u"]:
        out[f"{s}-S4"] = crash[crash.x_stage.eq(s)]
    for a, b in [("S1", "S2i"), ("S1", "S2u"), ("S2i", "S2u")]:
        out[f"{a}-{b}"] = p[(p.x_stage.eq(a) & p.y_stage.eq(b)) | (p.x_stage.eq(b) & p.y_stage.eq(a))]
    return out


def pool_pairs(pairs):
    res = {}
    for k, g in pairs.items():
        r = fit(g)
        if r is not None:
            res[k] = dict(k=r["k"], m=r["m"], z=r["b"][0], se=r["se"][0], r=np.tanh(r["b"][0]),
                          lo=np.tanh(r["lo"][0]), hi=np.tanh(r["hi"][0]), p=r["p"][0])
    return res


def matrix(zs):
    R = np.eye(5)
    for k, z in zs.items():
        a, b = k.split("-"); R[idx[a], idx[b]] = R[idx[b], idx[a]] = np.tanh(z)
    return R


def mediation(R, x):
    i, m, y = idx[x], idx["S3"], idx["S4"]
    a = R[i, m]
    A = np.array([[1, R[i, m]], [R[i, m], 1]]); bb = np.linalg.solve(A, [R[i, y], R[m, y]])
    direct, b = bb
    ind = a * b
    return dict(a=a, b=b, direct=direct, indirect=ind, total=ind + direct, prop=ind / (ind + direct))


def run(label, crash_codes, rng):
    res = pool_pairs(stage_pairs(D, crash_codes))
    tab = pd.DataFrame(res).T
    print(f"\n== {label}: stage-level pooled correlations"); print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(os.path.join(OUT, f"rq5_stage_pairs_{label}.csv"))
    rows = []
    for x in ["S1", "S2i", "S2u"]:
        need = [f"{x}-S3", "S3-S4", f"{x}-S4"]
        if not all(n in res for n in need):
            continue
        est = mediation(matrix({n: res[n]["z"] for n in need}), x)
        draws = pd.DataFrame([mediation(matrix({n: rng.normal(res[n]["z"], res[n]["se"]) for n in need}), x)
                              for _ in range(5000)])
        row = {"stage": x}
        for kx in ["a", "b", "direct", "indirect", "total", "prop"]:
            row[kx] = est[kx]; row[kx + "_lo"] = draws[kx].quantile(.025); row[kx + "_hi"] = draws[kx].quantile(.975)
        rows.append(row)
    mo = pd.DataFrame(rows)
    print(f"-- {label}: mediation"); print(mo.to_string(float_format=lambda v: f"{v:.3f}"))
    mo.to_csv(os.path.join(OUT, f"rq5_mediation_{label}.csv"), index=False)


def run_same_studies(rng):
    """Check: refit each process model with only the studies that report all three correlations."""
    pairs = stage_pairs(D, ["CRASH_SELF"])
    rows = []
    for x in ["S2i", "S2u"]:
        need = [f"{x}-S3", "S3-S4", f"{x}-S4"]
        same = set.intersection(*[set(pairs[n].cluster) for n in need])
        res = pool_pairs({n: pairs[n][pairs[n].cluster.isin(same)] for n in need})
        est = mediation(matrix({n: res[n]["z"] for n in need}), x)
        draws = pd.DataFrame([mediation(matrix({n: rng.normal(res[n]["z"], res[n]["se"]) for n in need}), x)
                              for _ in range(5000)])
        row = {"stage": x, "studies": len(same)}
        for kx in ["a", "b", "direct", "indirect", "total", "prop"]:
            row[kx] = est[kx]; row[kx + "_lo"] = draws[kx].quantile(.025); row[kx + "_hi"] = draws[kx].quantile(.975)
        rows.append(row)
    mo = pd.DataFrame(rows)
    print("-- same studies: mediation"); print(mo.to_string(float_format=lambda v: f"{v:.3f}"))
    mo.to_csv(os.path.join(OUT, "rq5_mediation_samestudies.csv"), index=False)


if __name__ == "__main__":
    D = load()
    rng = np.random.default_rng(20261023)
    run("selfreport", ["CRASH_SELF"], rng)
    run("recorded", ["CRASH_REC"], rng)
    run_same_studies(np.random.default_rng(20261026))
