# Figures 5-9 for the five RQs (one figure per RQ), computed from the extracted effects (run_rq_meta.load), the vote counts
# (vote_counts.py -> results/vote_counts.csv) and the Stage 3 frequency results.
# Figure 5: RQ1 (pooled r, moderators, and direction of effect for factors with fewer than four pooled studies); Figure 6: RQ2;
# Figure 7: RQ3, frequency, co-occurrence, and within-sample comparison; Figure 8: RQ4, pooled ORs and direction of effect;
# Figure 9: RQ5. Forest panels draw only significant pooled results. Run: python make_results_figs.py
# Sizes are the print sizes: 190 mm (7.48 in) for Figures 5, 7, 9 and 140 mm (5.51 in) for Figures 6, 8. All statistics have
# three decimals, rounded half up. Explanations go in the captions, never inside the figures.
import json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_rq_meta import load, pooled

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
FS, FST = 7.5, 8.5  # text and panel title sizes in points
plt.rcParams.update({"font.family": "Arial", "font.size": FS, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.width": 0.6, "ytick.major.size": 2.5, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.bf": "Arial:bold"})
W_FULL, W_MID = 7.48, 5.51


def r3(v):
    """Three decimals, rounded half up; the same rule as the text and Table S3."""
    return str(Decimal(repr(float(v))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).replace("-", "\u2212")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=1000, facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".svg"))
    plt.close(fig)


def forest(ax, rows, xlim, xticks, xlabel, log=False, ref=0.0, label_x=None, indent=0.0, val_x=None, k_x=None):
    """rows: list of ('head', text) or ('row', label, est, lo, hi, k, is_summary)."""
    y = 0
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    for r in rows:
        if r[0] == "gap":
            y -= 0.45
            continue
        if r[0] == "head":
            ax.text(label_x + indent / 2, y, r[1], fontsize=FS, style="italic", va="center", ha="left", transform=tr)
            y -= 0.7
            continue
        _, lab, est, lo, hi, k, summ = r[:7]
        mfc = r[7] if len(r) > 7 else "black"
        if summ:
            h = 0.3
            ax.fill([lo, est, hi, est], [y, y + h, y, y - h], color="black")
        else:
            ax.plot([lo, hi], [y, y], color="black", lw=0.7)
            ax.plot(est, y, marker="s", ms=3.4, mew=0.6, mfc=mfc, mec="black", color="black")
        ax.text(label_x + (0 if summ else indent), y, lab, fontsize=FS, va="center", ha="left",
                fontweight="bold" if summ else "normal", transform=tr)
        ax.text(val_x, y, f"{r3(est)} [{r3(lo)}, {r3(hi)}]", fontsize=FS, va="center", ha="left", transform=tr)
        ax.text(k_x, y, str(k), fontsize=FS, va="center", ha="right", transform=tr)
        y -= 0.8
    ax.axvline(ref, color="black", lw=0.5, ls=":")
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(y + 0.2, 0.6)
    ax.set_xticks(xticks); ax.set_xticklabels([f"{t:g}" for t in xticks])
    ax.minorticks_off()
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=1.5)
    return y


def pfmt(p):
    if p is None or p != p:
        return ""
    return "< 0.001" if p < 0.001 else r3(p)


def direction_panel(ax, rows, left_lab, right_lab, label_x, indent, unc_x, p_x, step, sign_lab="Sign test p"):
    """Diverging bars: studies in the risk direction (black, right) and the opposite direction (white, left).
    rows: ('head', text) or ('vote', label, more, unclear, fewer, p)."""
    tr = ax.get_yaxis_transform()
    M = max(max(r[2], r[4]) for r in rows if r[0] == "vote") + 1
    y = 0
    for r in rows:
        if r[0] == "head":
            ax.text(label_x + indent / 2, y, r[1], fontsize=FS, style="italic", va="center", ha="left", transform=tr)
            y -= 0.7
            continue
        _, lab, more, unc, few, p = r
        ax.barh(y, more, height=0.5, color="black", edgecolor="black", lw=0.6)
        ax.barh(y, -few, height=0.5, color="white", edgecolor="black", lw=0.6)
        if more:
            ax.text(more + M * 0.04, y, str(more), fontsize=FS, va="center", ha="left")
        if few:
            ax.text(-few - M * 0.04, y, str(few), fontsize=FS, va="center", ha="right")
        ax.text(label_x + indent, y, lab, fontsize=FS, va="center", ha="left", transform=tr)
        ax.text(unc_x, y, str(unc), fontsize=FS, va="center", ha="center", transform=tr)
        ax.text(p_x, y, pfmt(p), fontsize=FS, va="center", ha="center", transform=tr)
        y -= 0.8
    ax.axvline(0, color="black", lw=0.6)
    # each side spans its own largest count, with room for the side header when that side is nearly empty
    left = max(max(r[4] for r in rows if r[0] == "vote") + 1, 0.3 * M)
    right = max(max(r[2] for r in rows if r[0] == "vote") + 1, 0.3 * M)
    ax.set_xlim(-left, right); ax.set_ylim(y + 0.2, 1.25)
    t = [v for v in range(-int(left), int(right) + 1) if v % step == 0]
    ax.set_xticks(t); ax.set_xticklabels([str(abs(v)) for v in t])
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Number of studies", labelpad=1.5)
    ax.text(-M * 0.03, 0.85, left_lab, fontsize=FS, ha="right", va="center", linespacing=1.0)
    ax.text(M * 0.03, 0.85, right_lab, fontsize=FS, ha="left", va="center", linespacing=1.0)
    ax.text(unc_x, 0.85, "No clear\ndifference", fontsize=FS, ha="center", va="center", transform=tr, linespacing=1.0)
    ax.text(p_x, 0.85, sign_lab, fontsize=FS, ha="center", va="center", transform=tr, linespacing=1.0)


# Inch-based layout: every panel = title (TITLE_H) + plot area sized by its rows + x-axis label (XLAB_H).
TITLE_H, XLAB_H, U = 0.34, 0.34, 0.18
DIR_TITLE_H = 0.2  # direction panels draw their column headers inside the axes
VAL_IN, GAP_IN, K_IN = 1.04, 0.06, 0.34  # value and study-count columns of the forest panels


def units(rows):
    return sum(0.7 if r[0] == "head" else 0.45 if r[0] == "gap" else 0.8 for r in rows)


def forest_h(rows, u=U):
    return TITLE_H + (units(rows) + 0.4) * u + XLAB_H


def direction_h(rows, u=U):
    return DIR_TITLE_H + (units(rows) + 1.05) * u + XLAB_H


def add_ax(fig, W, H, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def forest_panel(fig, W, H, x, top, rows, title, lab_w, ax_w, xlim, xticks, xlabel, val_head, log=False, ref=0.0):
    h = (units(rows) + 0.4) * U
    ax = add_ax(fig, W, H, x + lab_w, top - TITLE_H - h, ax_w, h)
    val_x = 1 + GAP_IN / ax_w
    k_x = 1 + (GAP_IN + VAL_IN + K_IN) / ax_w
    forest(ax, rows, xlim, xticks, xlabel, log=log, ref=ref, label_x=-lab_w / ax_w, indent=0.06 / ax_w, val_x=val_x, k_x=k_x)
    tr = ax.get_yaxis_transform()
    ax.text(val_x, 0.75, val_head, fontsize=FS, ha="left", va="bottom", transform=tr)
    ax.text(k_x, 0.75, "Studies", fontsize=FS, ha="right", va="bottom", transform=tr)
    fig.text(x / W, top / H, title, fontsize=FST, va="top")


def direction_panel_at(fig, W, H, x, top, rows, title, lab_w, ax_w, left_lab, right_lab, unc_in, p_in, step, u=U, sign_lab="Sign test p"):
    h = (units(rows) + 1.05) * u
    ax = add_ax(fig, W, H, x + lab_w, top - DIR_TITLE_H - h, ax_w, h)
    direction_panel(ax, rows, left_lab, right_lab, -lab_w / ax_w, 0.06 / ax_w, 1 + unc_in / ax_w, 1 + p_in / ax_w, step, sign_lab)
    fig.text(x / W, top / H, title, fontsize=FST, va="top")


def class_rows(stage):
    out = []
    for sl, fn, mk, fc in SUBS:
        r = pooled(fn(BD[BD.x_stage.eq(stage)])); log.append(dict(label=f"{stage} / {sl}", **r))
        out.append(("row", sl, r["est"], r["lo"], r["hi"], r["k"], False, fc))
    return out


# ---------------------------------------------------------------- data
D = load()
P = D[D.primary_ok]
P = P[~(P.study_id.eq("Steinhauser2018") & P.y_label.str.contains("tracking", case=False))]
B = P[P.y_stage.eq("S3") & P.zero]
BD = B[~B.no_dir]
log = []


def row(label, g, metric="z", summ=False):
    r = pooled(g, metric)
    log.append(dict(label=label, **r))
    return ("row", label, r["est"], r["lo"], r["hi"], r["k"], summ)


VOTES = pd.read_csv(os.path.join(HERE, "results", "vote_counts.csv"))


def vote(rq, stage, code, label):
    v = VOTES[VOTES.rq.eq(rq) & VOTES.stage.eq(stage) & VOTES.factor.eq(code)].iloc[0]
    p = None if pd.isna(v.sign_p) else float(v.sign_p)
    log.append(dict(label=f"vote {rq} {stage} {label}", k=int(v.studies), m=np.nan, est=int(v.more), lo=int(v.unclear),
                    hi=int(v.fewer), p=p))
    return ("vote", label, int(v.more), int(v.unclear), int(v.fewer), p)


# ---------------------------------------------------------------- Figures 5 (RQ1) and 6 (RQ2)
SUBS = [("Violative behaviors", lambda g: g[g.y_code.eq("V")], "s", "black"),
        ("Non-violative behaviors", lambda g: g[g.y_code.eq("NV")], "s", "white")]


R_AX = dict(xlim=(-0.05, 0.7), xticks=[0, 0.2, 0.4, 0.6], xlabel="Pooled r with PAD behaviors", val_head="r [95% CI]")
rows5 = [row("All triggering factors", BD[BD.x_stage.eq("S1")], summ=True),
         ("head", "Individual factors"),
         row("Younger age", B[B.x_code.eq("AGE")]),
         row("Male gender", B[B.x_code.eq("MALE")]),
         row("Crash or violation history", B[B.x_code.eq("HISTORY")]),
         row("More driving exposure", B[B.x_code.eq("EXPOSURE")]),
         ("head", "Other traffic factors"),
         row("Social norms approving risky driving", B[B.x_code.eq("SOCIAL_NORMS")])]
vrows5 = [("head", "Individual factors"),
          vote("RQ1", "S1", "ALCOHOL_DRUG", "Alcohol or drug use"),
          vote("RQ1", "S1", "FAMILY", "Married or having children"),
          ("head", "Environmental factors"),
          vote("RQ1", "S1", "CONGESTION", "Congestion"),
          vote("RQ1", "S1", "TIME_PRESSURE", "Time pressure"),
          vote("RQ1", "S1", "NIGHT", "Night driving"),
          vote("RQ1", "S1", "PEAK", "Peak hours"),
          vote("RQ1", "S1", "WEEKEND", "Weekends"),
          vote("RQ1", "S1", "INTERSECTION", "Intersections"),
          vote("RQ1", "S1", "ROAD_SPEED", "Higher-speed roads"),
          vote("RQ1", "S1", "WEATHER", "Adverse weather"),
          ("head", "Vehicle factors"),
          vote("RQ1", "S1", "VEH_LARGE", "Larger vehicle"),
          vote("RQ1", "S1", "VEH_AGE", "Older vehicle"),
          ("head", "Other traffic factors"),
          vote("RQ1", "S1", "WORK", "Heavier work demands"),
          vote("RQ1", "S1", "PASSENGERS", "Passengers")]
brows5 = class_rows("S1")
# Figure 5: left column (a) above (b); right column (c)
TOP = 0.03
W = W_FULL
hl = forest_h(rows5) + forest_h(brows5)
hr = direction_h(vrows5)
H = max(hl, hr) + TOP
fig = plt.figure(figsize=(W, H))
top = H - TOP
forest_panel(fig, W, H, 0.03, top, rows5, "(a) Triggering factors and PAD behaviors", 1.82, 0.85, **R_AX)
forest_panel(fig, W, H, 0.03, forest_h(brows5), brows5, "(b) By behavior class", 1.82, 0.85, **R_AX)
direction_panel_at(fig, W, H, 4.2, top, vrows5, "(c) Factors synthesized by direction of effect", 1.36, 1.1,
                   "Fewer PAD\nbehaviors", "More PAD\nbehaviors", 0.25, 0.62, 2, sign_lab="Sign\ntest p")
save(fig, "fig5_rq1_triggering")

rows6 = [row("All intentional processes", BD[BD.x_stage.eq("S2i")], summ=True),
         row("Anger", B[B.x_code.eq("ANGER")]),
         row("Lower agreeableness", B[B.x_code.eq("PERS_A")]),
         row("Attitudes favoring risk", B[B.x_code.eq("ATTITUDE")]),
         row("Normlessness", B[B.x_sub.eq("NORMLESSNESS")]),
         row("Sensation seeking", B[B.x_code.eq("SENSATION")]),
         row("Poorer emotion regulation", B[B.x_code.eq("EMOTION_REG")]),
         row("Impulsivity", B[B.x_code.eq("IMPULSIVITY")]),
         ("gap",),
         row("All unintentional processes$^{\\mathbf{a}}$", BD[BD.x_stage.eq("S2u")], summ=True)]
brows6 = class_rows("S2i")
# Figure 6: one column, (a) above (b)
W = W_MID
H = forest_h(rows6) + forest_h(brows6) + TOP
fig = plt.figure(figsize=(W, H))
forest_panel(fig, W, H, 0.03, H - TOP, rows6, "(a) Psychological processes and PAD behaviors", 1.6, 2.4, **R_AX)
forest_panel(fig, W, H, 0.03, forest_h(brows6), brows6, "(b) Intentional processes by behavior class", 1.6, 2.4, **R_AX)
save(fig, "fig6_rq2_processes")

# ---------------------------------------------------------------- Figure 7 (RQ3)
vnv = json.load(open(os.path.join(HERE, "rq3_frequency", "cooccurrence_results.json"), encoding="utf-8"))
# item frequencies from the manuscript's frequency analysis (close following is violative; 6-8 studies per item)
freq = json.load(open(os.path.join(HERE, "rq3_frequency", "item_frequency_results.json"), encoding="utf-8"))
items = pd.DataFrame(freq["dbq"])
items = items[items.label.ne("Angry pursuit")]  # angry pursuit is not shown
for it in freq["dbq"]:
    log.append(dict(label="DBQ " + it["label"], k=it["studies"], m=it["k"], est=it["mean"], lo=it["ci_low"], hi=it["ci_high"], p=np.nan))
names = {"Speeding": "Speeding", "Red-light running": "Red-light running", "Unsafe lane changes": "Unsafe lane changes",
         "Illegal overtaking": "Illegal overtaking", "Racing": "Racing", "Close following": "Close following",
         "Horn use": "Horn use", "Hostile expression": "Hostile expression", "Angry pursuit": "Angry pursuit"}
items = items.sort_values("mean")
vrows7 = [("vote", "Within-sample\ncomparisons", 10, 6, 14, 0.5412561893463135)]  # rq3_frequency/within_sample_comparison_summary.json (30 studies)
log.append(dict(label="vote RQ3 within-sample (NV more, unclear, V more)", k=30, m=np.nan, est=10, lo=6, hi=14, p=0.5412561893463135))
# Figure 7: left column (a) above (c); right column (b) over the full height
W = W_FULL
UC7 = 0.22
A_TITLE, A_H, A_XLAB = 0.2, 0.17 * len(items), 0.52
H = TOP + A_TITLE + A_H + A_XLAB + direction_h(vrows7, UC7) + 0.04
fig = plt.figure(figsize=(W, H))
a1_bottom = H - TOP - A_TITLE - A_H
a1 = add_ax(fig, W, H, 1.1, a1_bottom, 2.1, A_H)
c_top = a1_bottom - A_XLAB
c_bottom = c_top - DIR_TITLE_H - (units(vrows7) + 1.05) * UC7
a2_bottom = c_bottom + 0.14  # room for the two-line axis title
a2 = add_ax(fig, W, H, 4.87, a2_bottom, 2.5, H - TOP - A_TITLE - a2_bottom)
for i, (_, it) in enumerate(items.iterrows()):
    a1.plot([it.ci_low, it.ci_high], [i, i], color="black", lw=0.7)
    a1.plot(it["mean"], i, marker="o", ms=3.8, mew=0.6, mfc="black" if it.cls == "V" else "white", mec="black")
a1.set_yticks(range(len(items))); a1.set_yticklabels([names[l] for l in items.label], fontsize=FS)
a1.tick_params(axis="y", length=0)
a1.set_xlim(0, 2); a1.set_xticks([0, 1, 2]); a1.set_xticklabels(["0\nnever", "1\nhardly ever", "2\noccasionally"], linespacing=1.0)
a1.set_ylim(-0.6, len(items) - 0.4)
for s in ["top", "right", "left"]:
    a1.spines[s].set_visible(False)
a1.set_xlabel("Pooled mean frequency (DBQ, 0\u20135)", labelpad=1.5)
a1.plot([], [], "o", ms=3.8, mew=0.6, mfc="black", mec="black", label="Violative"); a1.plot([], [], "o", ms=3.8, mew=0.6, mfc="white", mec="black", label="Non-violative")
a1.legend(loc="lower right", frameon=False, fontsize=FS, borderaxespad=0.1, handletextpad=0.2)
fig.text(0.03 / W, (H - TOP) / H, "(a) Frequency of PAD behaviors", fontsize=FST, va="top")
co = pd.DataFrame(vnv["cooccurrence_rows"])
co = co[co.id.ne("Yousaf2024_CN")].sort_values("r")  # same exclusion as the main analyses
from meta_core import ml_meta
zc, vc = np.arctanh(co.r.values), 1 / (co.n.values - 3)
mc = ml_meta(zc, vc, co.id.str.replace("_PK", "").values)
c = dict(k=len(co), r=np.tanh(mc["b"][0]), ci_low=np.tanh(mc["lo"][0]), ci_high=np.tanh(mc["hi"][0]))
log.append(dict(label="V-NV co-occurrence", k=mc["k"], m=mc["m"], est=c["r"], lo=c["ci_low"], hi=c["ci_high"], p=mc["p"][0]))
lab = {r.id: r.label.replace(" & ", " and ") for r in co.itertuples()}  # same form as the in-text citations
for i, r in enumerate(co.itertuples()):
    a2.plot([r.ci_low, r.ci_high], [i + 1.6, i + 1.6], color="black", lw=0.7)
    a2.plot(r.r, i + 1.6, marker="s", ms=3.4, color="black")
a2.fill([c["ci_low"], c["r"], c["ci_high"], c["r"]], [0, 0.32, 0, -0.32], color="black")
a2.set_yticks([0] + [i + 1.6 for i in range(len(co))])
a2.set_yticklabels([f"Pooled, {c['k']} samples"] + [lab[i] for i in co.id], fontsize=FS)
a2.tick_params(axis="y", length=0)
a2.set_xlim(0, 1); a2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]); a2.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
for s in ["top", "right", "left"]:
    a2.spines[s].set_visible(False)
a2.text(c["ci_low"] - 0.02, 0, f"{r3(c['r'])} [{r3(c['ci_low'])}, {r3(c['ci_high'])}]", fontsize=FS, va="center", ha="right")
a2.set_xlabel("Correlation between violative and\nnon-violative behavior scales", labelpad=1.5, linespacing=1.0)
a2.set_ylim(-0.6, len(co) + 1.0)
fig.text(3.45 / W, (H - TOP) / H, "(b) Co-occurrence of violative and non-violative behaviors", fontsize=FST, va="top")
direction_panel_at(fig, W, H, 0.03, c_top, vrows7, "(c) Within-sample comparison of frequency", 0.88, 1.22,
                   "Violative\nmore frequent", "Non-violative\nmore frequent", 0.42, 0.92, 5, u=UC7)
save(fig, "fig7_rq3_expression")

# ---------------------------------------------------------------- Figure 8 (RQ4)
O = P[P.x_stage.eq("S3") & P.y_stage.eq("S4")]
OC = O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]
NVC = ["NV_SCALE", "HORN", "GESTURE", "PURSUIT"]
VC = ["V_SCALE", "SPEED", "FOLLOW", "LANE", "OVERTAKE", "SIGNAL", "RACE"]
OI = O[O.y_code.eq("INJURY")]
rows8 = [row("Crash involvement, all behaviors", OC, "logor", summ=True),
         row("Violative behaviors", OC[OC.x_code.isin(VC)], "logor"),
         row("Non-violative behaviors", OC[OC.x_code.isin(NVC)], "logor"),
         ("gap",),
         row("Injury in crashes, all behaviors", OI, "logor", summ=True),
         row("Aggressive or reckless driving code", OI[OI.x_code.eq("AGG_CODE")], "logor")]
vrows8 = [("head", "Crash involvement"),
          vote("RQ4", "crash", "violative", "Violative behaviors"),
          vote("RQ4", "crash", "non-violative", "Non-violative behaviors"),
          vote("RQ4", "crash", "mixed", "Mixed behavior scales"),
          ("head", "Injury in crashes"),
          vote("RQ4", "injury", "violative", "Violative behaviors"),
          vote("RQ4", "injury", "aggressive code", "Aggressive or reckless driving code"),
          ("head", "Severe or fatal injury"),
          vote("RQ4", "severe", "violative", "Violative behaviors"),
          vote("RQ4", "severe", "aggressive code", "Aggressive or reckless driving code")]
# Figure 8: one column, (a) above (b)
W = W_MID
H = forest_h(rows8) + direction_h(vrows8) + TOP
fig = plt.figure(figsize=(W, H))
forest_panel(fig, W, H, 0.03, H - TOP, rows8, "(a) Pooled odds ratios", 1.86, 2.14, (0.8, 5), [1, 1.5, 2, 3, 5],
             "Pooled odds ratio (log scale)", "OR [95% CI]", log=True, ref=1.0)
direction_panel_at(fig, W, H, 0.03, direction_h(vrows8), vrows8, "(b) Direction of effect in all studies", 1.86, 2.14,
                   "Lower risk", "Higher risk", 0.44, 1.02, 10)
save(fig, "fig8_rq4_outcomes")

# ---------------------------------------------------------------- Figure 9 (RQ5): path diagrams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
med = pd.read_csv(os.path.join(HERE, "results", "rq5_mediation_selfreport.csv")).set_index("stage")
# path a is the pooled correlation of the process with PAD behaviors, so it carries the same CI as Figure 6
res = pd.read_csv(os.path.join(HERE, "results", "results_table.csv")).set_index("label")
for st, lab_ in [("S2i", "Intentional processes"), ("S2u", "Unintentional processes")]:
    med.loc[st, ["a", "a_lo", "a_hi"]] = res.loc[lab_, ["est", "lo", "hi"]].values


def ci(row, k):
    return f"{r3(row[k])} [{r3(row[k + '_lo'])}, {r3(row[k + '_hi'])}]"


def path_panel(ax, row, proc, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = {"x": (0.16, 0.36, proc), "m": (0.50, 0.76, "PAD behaviors"), "y": (0.84, 0.36, "Self-reported\ncrash involvement")}
    w, h = 0.27, 0.17
    for cx, cy, lab in boxes.values():
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0", fc="white", ec="black", lw=0.8))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=FS, linespacing=1.0)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black", shrinkA=0, shrinkB=0))
    arrow((0.23, 0.45), (0.40, 0.675))
    arrow((0.60, 0.675), (0.77, 0.45))
    arrow((0.295, 0.36), (0.705, 0.36))
    ax.text(0.30, 0.60, ci(row, "a"), ha="right", va="center", fontsize=FS)
    ax.text(0.70, 0.60, ci(row, "b"), ha="left", va="center", fontsize=FS)
    ax.text(0.50, 0.335, "Direct " + ci(row, "direct"), ha="center", va="top", fontsize=FS)
    ax.text(0.50, 0.09, f"Indirect through PAD behaviors {ci(row, 'indirect')}\n{row['prop'] * 100:.0f}% of the total {r3(row['total'])}",
            ha="center", va="center", fontsize=FS, linespacing=1.15)
    ax.text(0.0, 1.0, title, ha="left", va="top", fontsize=FST, transform=ax.transAxes)


fig, (p1, p2) = plt.subplots(1, 2, figsize=(W_FULL, 2.3))
path_panel(p1, med.loc["S2i"], "Intentional\nprocesses", "(a) Intentional processes")
path_panel(p2, med.loc["S2u"], "Unintentional\nprocesses", "(b) Unintentional processes")
fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.03)
save(fig, "fig9_rq5_chain")

pd.DataFrame(log).to_csv(os.path.join(FIG, "figure_values.csv"), index=False)
print(pd.DataFrame(log)[["label", "k", "m", "est", "lo", "hi", "p"]].to_string(float_format=lambda v: f"{v:.3f}"))
