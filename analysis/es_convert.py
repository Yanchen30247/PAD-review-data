# Convert extracted raw statistics (PROTOCOL.md schema) to risk-aligned r, Fisher z and log OR.
# Formulas follow Borenstein et al. (2009, ch. 7): d from means/SD, t, F(1); phi from chi-square(1);
# log OR from 2x2 counts or printed OR/CI; d = log OR * sqrt(3)/pi; r = d / sqrt(d^2 + a), a = (n1+n2)^2/(n1 n2).
import math
import numpy as np
import pandas as pd

K = math.sqrt(3) / math.pi


def _f(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return np.nan
        return float(str(x).replace("−", "-"))
    except ValueError:
        return np.nan


def _a(n1, n2):
    return (n1 + n2) ** 2 / (n1 * n2) if n1 > 0 and n2 > 0 else 4.0


def _dir(row):
    d = str(row.get("direction", "")).strip()
    return -1.0 if d in ("-", "−") else 1.0


def convert(row):
    """Return dict with r (aligned), z, vz, logor (aligned), vlogor, n, method; NaN when not convertible."""
    st = str(row.get("stat_type", "")).strip()
    sx = _f(row.get("sign_x")); sy = _f(row.get("sign_y"))
    sx = 1.0 if np.isnan(sx) else sx; sy = 1.0 if np.isnan(sy) else sy
    sgn = sx * sy
    n1, n2, n = _f(row.get("n1")), _f(row.get("n2")), _f(row.get("n_total"))
    if np.isnan(n) and not (np.isnan(n1) or np.isnan(n2)):
        n = n1 + n2
    d = np.nan; r = np.nan; logor = np.nan; vlogor = np.nan; method = st
    if st == "r":
        r = _f(row.get("r_value")) * sgn
    elif st == "means_sd":
        m1, s1, m2, s2 = (_f(row.get(c)) for c in ("m1", "sd1", "m2", "sd2"))
        if not any(np.isnan(x) for x in (m1, s1, m2, s2, n1, n2)) and n1 + n2 > 2:
            sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
            d = (m1 - m2) / sp * sgn if sp > 0 else np.nan
    elif st == "d":
        d = _f(row.get("d_value")) * sgn
    elif st in ("t_ind", "F1"):
        # F has no sign: its sign comes from the aligned direction column, so it is not multiplied by sign_x * sign_y again
        t = _f(row.get("t_value")) if st == "t_ind" else math.sqrt(abs(_f(row.get("F_value")))) * _dir(row)
        if st == "F1":
            sgn = 1.0
        if not np.isnan(t):
            if not (np.isnan(n1) or np.isnan(n2)):
                d = t * math.sqrt(1 / n1 + 1 / n2) * sgn
            else:
                df = _f(row.get("df")); df = n - 2 if np.isnan(df) else df
                r = t / math.sqrt(t ** 2 + df) * sgn if df > 0 else np.nan
    elif st == "t_paired":
        t = _f(row.get("t_value")); df = _f(row.get("df")); df = n - 1 if np.isnan(df) else df
        r = t / math.sqrt(t ** 2 + df) * sgn if df > 0 and not np.isnan(t) else np.nan
    elif st == "chi2_1df":
        c2 = _f(row.get("chi2"))
        if not np.isnan(c2) and n > 0:
            r = math.sqrt(c2 / n) * _dir(row)
    elif st in ("table_2x2", "prop_2groups"):
        if st == "prop_2groups":
            p1, p2 = _f(row.get("p1")), _f(row.get("p2"))
            if p1 > 1 or p2 > 1:
                p1, p2 = p1 / 100, p2 / 100
            a, c = p1 * n1, p2 * n2; b, dd = n1 - a, n2 - c
        else:
            a, b, c, dd = (_f(row.get(k)) for k in ("a", "b", "c", "d_cell"))
            n1, n2 = a + b, c + dd; n = n1 + n2
        if not any(np.isnan(x) for x in (a, b, c, dd)):
            if min(a, b, c, dd) == 0:
                a, b, c, dd = a + .5, b + .5, c + .5, dd + .5
            logor = math.log(a * dd / (b * c)) * sgn
            vlogor = 1 / a + 1 / b + 1 / c + 1 / dd
            d = logor * K
    elif st in ("OR_crude", "OR_adj", "RR_crude"):
        o, lo, hi = _f(row.get("or_value")), _f(row.get("or_lo")), _f(row.get("or_hi"))
        if o > 0:
            logor = math.log(o) * sgn
            if lo > 0 and hi > 0:
                vlogor = ((math.log(hi) - math.log(lo)) / 3.92) ** 2
            d = logor * K
    elif st == "z_mw":
        z = _f(row.get("z_value")) if "z_value" in row else _f(row.get("t_value"))
        if not np.isnan(z) and n > 0:
            r = z / math.sqrt(n) * sgn
    # fill the other metrics
    if np.isnan(r) and not np.isnan(d):
        r = d / math.sqrt(d ** 2 + _a(n1 if not np.isnan(n1) else 0, n2 if not np.isnan(n2) else 0))
    if np.isnan(logor) and not np.isnan(r) and abs(r) < 1:
        dd_ = 2 * r / math.sqrt(1 - r ** 2)
        logor = dd_ / K
        if n > 3:
            vr = (1 - r ** 2) ** 2 / (n - 1)
            vlogor = 4 * vr / (1 - r ** 2) ** 3 / K ** 2
    if not np.isnan(r):
        r = max(min(r, 0.995), -0.995)
    z = math.atanh(r) if not np.isnan(r) else np.nan
    vz = 1 / (n - 3) if n > 3 else np.nan
    ratio_based = st in ("table_2x2", "prop_2groups", "OR_crude", "OR_adj", "RR_crude")
    if (np.isnan(vz) or ratio_based) and not np.isnan(vlogor) and not np.isnan(r):
        # variance of z from the log OR variance: v_d = v_logOR * 3/pi^2, v_r = a^2 v_d / (d^2 + a)^3 (a = 4), v_z = v_r / (1 - r^2)^2
        vd = vlogor * K ** 2; dd_ = 2 * r / math.sqrt(1 - r ** 2)
        vz = 16 * vd / (dd_ ** 2 + 4) ** 3 / (1 - r ** 2) ** 2
    return dict(r=r, z=z, vz=vz, logor=logor, vlogor=vlogor, n_eff=n, conv=method)


def convert_frame(df):
    out = pd.DataFrame([convert(r) for _, r in df.iterrows()], index=df.index)
    return pd.concat([df, out], axis=1)
