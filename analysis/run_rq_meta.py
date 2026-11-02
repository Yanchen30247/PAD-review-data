# Pool the extracted effects (es_extract/*/effects_*.csv) for RQ1, RQ2, RQ4 and RQ5.
# Effects are converted with es_convert.py (risk-aligned r, Fisher z, log OR) and pooled with three-level
# random-effects models (effects nested in studies; meta_core.ml_meta). Moderators: measurement source of the
# behavior or outcome (self-report vs objective), behavior class (V vs NV), design (experiment vs correlational).
# Run: python run_rq_meta.py
import glob, os, re
import numpy as np, pandas as pd
from es_convert import convert_frame
from meta_core import ml_meta
import corrections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
NO_DIRECTION = {"EDUCATION", "INCOME", "EMPLOYED", "OCCUPATION", "FAMILY", "LOCATION", "ROAD", "ROAD_SPEED",
                "INTERSECTION", "VEHICLE", "VEH_AGE", "VEH_STATUS", "INTERVENTION", "EXT_OTHER", "BG_OTHER",
                "PERS_E", "PERS_O", "NO_BELT"}
F = lambda x: f"{x:.3f}"


def load():
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                   for f in glob.glob(os.path.join(HERE, "*", "effects_*.csv"))], ignore_index=True)
    for c in d.columns:
        d[c] = d[c].str.strip()
    d = corrections.apply(d)
    d["x_code"] = d.x_code.replace({"GENDER": "MALE"})
    note = d.note.str.lower()
    d["sens_only"] = note.str.contains(r"sensitivity[ _-]only")
    d["doubt"] = note.str.contains("doubt_not_behavior")
    d["no_dir"] = d.x_code.isin(NO_DIRECTION) | note.str.contains("no a priori direction")
    d["same_instr"] = note.str.contains("same instrument")
    d["extreme"] = note.str.contains("extreme group")
    d["lifetime"] = note.str.contains("lifetime")
    d = d[~d.x_code.isin(["KINEMATIC", "SIM_COLLISION"]) & ~d.y_code.isin(["SIM_COLLISION"])]
    # B1 and B2 both coded some MegiasRobles2022 and normlessness rows: keep the first agent's row
    d = d.sort_values("agent")
    has_id = d.row_id.ne("")
    dup = has_id & d.duplicated(["study_id", "row_id", "x_code", "y_code", "r_value", "or_value", "m1"], keep="first") \
        & d.groupby(["study_id", "row_id", "x_code", "y_code"]).agent.transform("nunique").gt(1)
    d = d[~dup]
    d = convert_frame(d)
    d["primary_ok"] = ~d.primary.str.lower().eq("no") & ~d.sens_only & ~d.doubt & d.r.notna()
    d["zero"] = ~d.zero_order.str.lower().eq("no") & ~d.adjusted.str.lower().eq("yes")
    d["latent_flag"] = d.latent.str.lower().eq("yes")
    d["obj_y"] = d.y_source.isin(["objective", "other_report"]).astype(float)
    d["obj_x"] = d.x_source.isin(["objective", "manipulated"]).astype(float)
    d["exp"] = d.design.str.startswith("experiment").astype(float)
    d["is_nv"] = d.y_code.eq("NV").astype(float)
    d["cluster"] = d.study_id
    return d


def fit(g, metric="z", X=None):
    v = g.vz if metric == "z" else g.vlogor
    ok = (g[metric].notna() & v.notna() & (v > 0)).values
    if g[ok].cluster.nunique() < 2:
        return None
    try:
        return ml_meta(g[metric].values[ok], v.values[ok], g.cluster.values[ok], None if X is None else X[ok])
    except np.linalg.LinAlgError:
        return None


def pooled(g, metric="z", mods=()):
    res = fit(g, metric)
    if res is None:
        return None
    tr = np.tanh if metric == "z" else np.exp
    row = dict(k=res["k"], m=res["m"], est=tr(res["b"][0]), lo=tr(res["lo"][0]), hi=tr(res["hi"][0]), p=res["p"][0],
               tau2_b=res["tau2_between"], tau2_w=res["tau2_within"], m_obj=int(g.obj_y.sum()))
    for c in mods:
        if g[c].nunique() < 2 or g.cluster.nunique() < 4:
            continue
        r2 = fit(g, metric, np.column_stack([np.ones(len(g)), g[c].values]))
        if r2 is not None:
            row[c + "_b"] = r2["b"][1]; row[c + "_p"] = r2["p"][1]
            if metric == "z":  # subgroup estimates implied by the moderator model
                row[c + "_est0"] = np.tanh(r2["b"][0]); row[c + "_est1"] = np.tanh(r2["b"][0] + r2["b"][1])
            else:
                row[c + "_est0"] = np.exp(r2["b"][0]); row[c + "_est1"] = np.exp(r2["b"][0] + r2["b"][1])
    return row


def table(d, keys, metric="z", mods=(), min_k=2):
    rows = []
    for key, g in d.groupby(keys):
        r = pooled(g, metric, mods)
        if r is None or r["k"] < min_k:
            continue
        rows.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), **r})
    return pd.DataFrame(rows)


def show(name, t):
    t.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    print(f"\n== {name}"); print(t.to_string(float_format=F) if len(t) else "(empty)")


if __name__ == "__main__":
    d = load()
    d.to_csv(os.path.join(OUT, "all_effects_converted.csv"), index=False, encoding="utf-8-sig")
    p = d[d.primary_ok]
    print("rows", len(d), "usable primary", len(p), "studies", p.study_id.nunique())

    # ---------- RQ1, RQ2: triggering factors and psychological processes with behaviors ----------
    b = p[p.y_stage.eq("S3") & p.zero & p.x_stage.isin(["S1", "S2i", "S2u"])]
    show("rq12_factor", table(b, ["x_stage", "x_code"], mods=("obj_y", "is_nv", "exp")))
    show("rq12_factor_class", table(b[b.y_code.isin(["V", "NV"])], ["x_stage", "x_code", "y_code"]))
    show("rq2_psy_other_sub", table(b[b.x_code.eq("PSY_OTHER")], ["x_sub"]))
    bd = b[~b.no_dir]
    show("rq12_stage", table(bd, ["x_stage"], mods=("obj_y", "is_nv", "exp")))
    show("rq12_stage_no_same_instrument", table(bd[~bd.same_instr & ~bd.latent_flag], ["x_stage"], mods=("obj_y",)))

    # ---------- RQ4: behaviors with safety outcomes (log OR) ----------
    o = p[p.x_stage.eq("S3") & p.y_stage.eq("S4")].copy()
    o["y_grp"] = o.y_code.replace({"CRASH_SELF": "CRASH", "CRASH_REC": "CRASH"})
    o["adj"] = o.adjusted.str.lower().eq("yes").astype(float)
    show("rq4_behavior", table(o, ["level", "y_grp", "x_code"], metric="logor", mods=("obj_y", "obj_x", "adj")))
    od = o[o.level.eq("driver") & o.y_grp.eq("CRASH")]
    od = od.assign(cls=np.where(od.x_code.isin(["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]), "NV",
                       np.where(od.x_code.isin(["MIXED_SCALE", "AGG_CODE"]), "Mixed", "V")))
    show("rq4_driver_crash_class", table(od, ["cls"], metric="logor", mods=("obj_y", "obj_x", "adj")))
    show("rq4_driver_crash_all", table(od.assign(all="all"), ["all"], metric="logor", mods=("obj_y", "obj_x", "adj")))
    show("rq4_injury_all", table(o[o.y_grp.isin(["INJURY", "SEVERE"])].assign(all="all"), ["y_grp"], metric="logor",
                                 mods=("adj",)))

    # ---------- RQ5 inputs: S1 and S2 with crash involvement (driver level) ----------
    c = p[p.x_stage.isin(["S1", "S2i", "S2u"]) & p.y_code.isin(["CRASH_SELF", "CRASH_REC"]) & p.level.isin(["driver", ""])
          & p.zero]
    show("rq5_factor_crash", table(c, ["x_stage", "x_code", "y_code"]))
    show("rq5_stage_crash", table(c[~c.no_dir], ["x_stage", "y_code"]))
