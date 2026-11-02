# Three-level random-effects meta-analysis and meta-regression (effects nested in clusters), REML,
# with cluster-robust (CR1) standard errors and t tests on (number of clusters - number of coefficients) df.
import numpy as np
from scipy import optimize, stats


def ml_meta(y, v, cluster, X=None):
    y = np.asarray(y, float); v = np.asarray(v, float); cluster = np.asarray(cluster)
    X = np.ones((len(y), 1)) if X is None else np.asarray(X, float)
    ids = list(dict.fromkeys(cluster))
    blocks = [np.where(cluster == c)[0] for c in ids]
    p = X.shape[1]

    def parts(lt):
        tb, tw = np.exp(lt)
        Wi, ld = [], 0.0
        for ix in blocks:
            m = len(ix)
            V = tb * np.ones((m, m)) + np.diag(tw + v[ix])
            Wi.append(np.linalg.inv(V)); ld += np.linalg.slogdet(V)[1]
        XtWX = sum(X[ix].T @ W @ X[ix] for ix, W in zip(blocks, Wi))
        XtWy = sum(X[ix].T @ W @ y[ix] for ix, W in zip(blocks, Wi))
        b = np.linalg.solve(XtWX, XtWy)
        return Wi, ld, XtWX, b

    def nll(lt):
        Wi, ld, XtWX, b = parts(lt)
        q = sum((y[ix] - X[ix] @ b) @ W @ (y[ix] - X[ix] @ b) for ix, W in zip(blocks, Wi))
        return 0.5 * (ld + np.linalg.slogdet(XtWX)[1] + q)

    best = None
    for x0 in ([-4.6, -5.3], [-2.3, -6.9], [-6.9, -2.3]):
        o = optimize.minimize(nll, x0=x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 3000})
        if best is None or o.fun < best.fun:
            best = o
    Wi, ld, XtWX, b = parts(best.x)
    inv = np.linalg.inv(XtWX)
    se_model = np.sqrt(np.diag(inv))
    k = len(blocks)
    meat = np.zeros((p, p))
    for ix, W in zip(blocks, Wi):
        u = X[ix].T @ W @ (y[ix] - X[ix] @ b)
        meat += np.outer(u, u)
    rob = inv @ meat @ inv * (k / (k - p) if k > p else np.nan)
    se_rob = np.sqrt(np.diag(rob))
    se = np.fmax(se_model, se_rob) if k > p else se_model
    df = max(k - p, 1)
    tq = stats.t.ppf(0.975, df)
    pval = 2 * stats.t.sf(np.abs(b / se), df)
    tb, tw = np.exp(best.x)
    return dict(b=b, se=se, lo=b - tq * se, hi=b + tq * se, p=pval, k=k, m=len(y), df=df,
                tau2_between=tb, tau2_within=tw, se_model=se_model, se_robust=se_rob)
