# Sensitivity analyses named in Methods 3.3.5, on the verified data:
# (1) adding the sensitivity-only samples back, (2) omitting latent correlations, (3) omitting same-questionnaire
# error/lapse-behavior effects, (4) omitting lifetime crash counts. Reported for the stage-level links and the RQ5 chain.
# Run: python sensitivity_runs.py
import os
import numpy as np, pandas as pd
from run_rq_meta import load, fit, OUT
import rq5_chain as ch

D0 = load()


def chain(d):
    pairs = ch.stage_pairs(d, ["CRASH_SELF"])
    res = ch.pool_pairs(pairs)
    rows = {}
    for k in ["S1-S3", "S2i-S3", "S2u-S3", "S3-S4", "S1-S4", "S2i-S4", "S2u-S4"]:
        if k in res:
            rows[k] = f"{res[k]['r']:.3f} [{res[k]['lo']:.3f}, {res[k]['hi']:.3f}] k={res[k]['k']}"
    for x in ["S1", "S2i", "S2u"]:
        need = [f"{x}-S3", "S3-S4", f"{x}-S4"]
        if all(n in res for n in need):
            e = ch.mediation(ch.matrix({n: res[n]["z"] for n in need}), x)
            rows[f"{x} prop"] = f"{e['prop']:.2f} (indirect {e['indirect']:.3f}, total {e['total']:.3f})"
    return rows


variants = {
    "primary": D0,
    "with sensitivity-only samples": D0.assign(primary_ok=(~D0.primary.str.lower().eq("no") & ~D0.doubt & D0.r.notna())),
    "without latent correlations": D0[~D0.latent_flag],
    "without same-questionnaire errors/lapses": D0[~D0.same_instr],
    "without lifetime crash counts": D0[~D0.lifetime],
}
out = {}
for name, d in variants.items():
    out[name] = chain(d)
tab = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(tab.to_string())
tab.to_csv(os.path.join(OUT, "sensitivity_runs.csv"))
