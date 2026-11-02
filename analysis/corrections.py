# Apply the second verification (verify/verify_V1-V3.csv) to the merged extraction, plus decisions on the rows the
# checkers marked unsure. Called by run_rq_meta.load() before conversion. Every rule is listed here so it can be audited.
import glob, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def apply(d):
    d = d.set_index("es_id", drop=False)
    log = []
    for f in sorted(glob.glob(os.path.join(HERE, "verify", "verify_V*.csv"))):
        v = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for _, r in v[v.verdict.eq("fix")].iterrows():
            if r.es_id not in d.index or r.field not in d.columns:
                log.append(("fix_skipped", r.es_id, r.field)); continue
            new = r.correct_value.strip().strip('"')
            if new.lower().startswith("append"):
                d.loc[r.es_id, r.field] = d.loc[r.es_id, r.field] + "; " + new[6:].strip().strip('"')
            else:
                d.loc[r.es_id, r.field] = "" if new == "(blank)" else new
            log.append(("fix", r.es_id, r.field))
        ex = v[v.verdict.eq("exclude")].es_id
        log += [("exclude", e, "verify") for e in ex if e in d.index]
        d = d.drop(index=[e for e in ex if e in d.index])

    lab = (d.x_label + " " + d.y_label + " " + d.note).str.lower()
    drop = (
        (d.study_id.eq("Sarma2013") & d.y_label.str.contains("reckless", case=False))            # mostly lapse items
        | (d.study_id.eq("Li2020") & ~d.y_label.str.contains("Velocity (?:mean|max)", case=False))  # vehicle-control kinematics
        | (d.study_id.eq("Deffenbacher2003") & d.y_code.eq("STATE"))                               # composition artifact
        | (d.study_id.eq("Steinhauser2018") & d.y_label.str.contains("variab|tracking", case=False))
        | d.x_label.str.contains("desirab", case=False)                                            # social desirability
    )
    log += [("exclude", e, "unsure rule") for e in d.index[drop]]
    d = d[~drop].copy()
    lab = (d.x_label + " " + d.y_label + " " + d.note).str.lower()
    # no defensible risk direction: keep out of stage-level pooling
    nodir = d.study_id.eq("Ge2021") | (d.study_id.eq("Love2024") & lab.str.contains("rank")) | lab.str.contains("enforcement")
    d.loc[nodir, "note"] = d.loc[nodir, "note"] + "; no a priori direction (verification)"
    # gender coding of Love2024 cannot be settled (Methods vs Results): sensitivity only
    lg = d.study_id.eq("Love2024") & d.x_code.isin(["MALE", "GENDER"])
    d.loc[lg, "note"] = d.loc[lg, "note"] + "; sensitivity-only: gender coding unresolved"
    # per-unit ratios of a continuous predictor are not two-group contrasts
    pu = lab.str.contains(r"per[- ]unit") & d.stat_type.isin(["OR_adj", "OR_crude", "RR_crude"])
    d.loc[pu, "stat_type"] = "not_convertible"
    # DAX aggressive use of the vehicle mixes V and NV items: code Mixed everywhere
    dax = d.y_label.str.contains("vehicle", case=False) & lab.str.contains("dax") & d.y_stage.eq("S3")
    d.loc[dax, "y_code"] = "Mixed"
    log += [("rule_nodir", e, "") for e in d.index[nodir]] + [("rule_perunit", e, "") for e in d.index[pu]]
    pd.DataFrame(log, columns=["action", "es_id", "detail"]).to_csv(os.path.join(HERE, "results", "corrections_log.csv"), index=False)
    return d.reset_index(drop=True)
