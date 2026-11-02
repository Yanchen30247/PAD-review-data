# Figure 10: funnel plots for the main pooled estimates (same subsets as results_table.py), with the multilevel Egger test
# (standard error as a moderator in the three-level model). Effect sizes on the analysis scale: Fisher z or log OR.
# Print size 190 mm (7.48 in); statistics have three decimals, rounded half up.
# Run: python make_funnels.py
import os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from run_rq_meta import load
from meta_core import ml_meta

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.width": 0.6, "ytick.major.size": 2.5, "pdf.fonttype": 42})

D = load()
P = D[D.primary_ok]
P = P[~(P.study_id.eq("Steinhauser2018") & P.y_label.str.contains("tracking", case=False))]
B = P[P.y_stage.eq("S3") & P.zero]
BD = B[~B.no_dir]
O = P[P.x_stage.eq("S3") & P.y_stage.eq("S4")]
OC = O[O.level.eq("driver") & O.y_code.isin(["CRASH_SELF", "CRASH_REC"])]
OI = O[O.y_code.eq("INJURY")]
C = P[P.y_code.eq("CRASH_SELF") & P.level.isin(["driver", ""]) & P.zero & ~P.no_dir]

PANELS = [("(a) Triggering factors", BD[BD.x_stage.eq("S1")], "z"),
          ("(b) Younger age", B[B.x_code.eq("AGE")], "z"),
          ("(c) Male gender", B[B.x_code.eq("MALE")], "z"),
          ("(d) Intentional processes", BD[BD.x_stage.eq("S2i")], "z"),
          ("(e) Unintentional processes", BD[BD.x_stage.eq("S2u")], "z"),
          ("(f) Crash involvement\n(odds ratio)", OC, "logor"),
          ("(g) Injury in crashes\n(odds ratio)", OI, "logor"),
          ("(h) Self-reported crash\ninvolvement (r)", C[C.x_stage.eq("S3")], "z")]


def pfmt(p):
    if p < 0.001:
        return "< 0.001"
    return "= " + str(Decimal(repr(float(p))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


fig, axes = plt.subplots(2, 4, figsize=(7.48, 4.3))
rows = []
for ax, (title, g, metric) in zip(axes.flat, PANELS):
    v = g.vz if metric == "z" else g.vlogor
    ok = g[metric].notna() & v.notna() & (v > 0)
    y, vv, cl = g[metric][ok].values, v[ok].values, g.cluster[ok].values
    se = np.sqrt(vv)
    r = ml_meta(y, vv, cl)
    eg = ml_meta(y, vv, cl, np.column_stack([np.ones(len(y)), se]))
    est = r["b"][0]
    rows.append(dict(panel=title.replace("\n", " "), k=r["k"], m=r["m"], est=est, egger_b=eg["b"][1], egger_p=eg["p"][1]))
    smax = se.max() * 1.05
    ss = np.linspace(0, smax, 50)
    ax.plot(est - 1.96 * ss, ss, color="black", lw=0.5, ls="--")
    ax.plot(est + 1.96 * ss, ss, color="black", lw=0.5, ls="--")
    ax.axvline(est, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.45, ls=":")
    ax.scatter(y, se, s=4.5, facecolors="none", edgecolors="black", linewidths=0.35, alpha=0.75)
    ax.set_ylim(smax, 0)
    lo = min(y.min(), est - 1.96 * smax); hi = max(y.max(), est + 1.96 * smax)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo - pad, hi + pad)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{title}\nEgger p {pfmt(eg['p'][1])}", loc="left", fontsize=7.5, pad=3, linespacing=1.05)
    # ticks on round steps only, so that no label is a rounded 0.025
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "−")))
    ax.set_xlabel("Log odds ratio" if metric == "logor" else "Fisher z", labelpad=1.5)
for ax in axes[:, 0]:
    ax.set_ylabel("Standard error", labelpad=2)
fig.subplots_adjust(left=0.062, right=0.985, top=0.915, bottom=0.075, wspace=0.34, hspace=0.62)
fig.savefig(os.path.join(FIG, "fig10_funnel.png"), dpi=1000, facecolor="white")
fig.savefig(os.path.join(FIG, "fig10_funnel.pdf"))
fig.savefig(os.path.join(FIG, "fig10_funnel.svg"))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(HERE, "results", "funnel_egger.csv"), index=False)
print(t.to_string(float_format=lambda x: f"{x:.4f}"))
