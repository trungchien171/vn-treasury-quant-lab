"""PCA of daily yield curve changes (level / slope / curvature)."""
import numpy as np
import pandas as pd

from .data import BREAKS, changes


def curve_pca(rates, n=3, horizon=1, breaks=BREAKS):
    """rates: date x tenor DataFrame of zero rates (decimals).

    Returns (explained variance ratio, loadings tenor x PC, scores date x PC) on
    `horizon`-day changes in basis points. Signs are normalised so PC1 (level) loads
    positively, PC2 (slope) rises with tenor and PC3 (curvature) is positive in the belly.
    """
    dx = changes(rates, horizon, breaks) * 1e4
    dx = dx.loc[:, dx.std() > 0]
    cov = np.cov(dx.values, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    tenors = dx.columns.to_numpy(float)
    signs = [np.sign(vecs[:, 0].sum()),
             np.sign(vecs[-1, 1] - vecs[0, 1]) if n > 1 else 1,
             np.sign(vecs[np.argmin(np.abs(tenors - 5)), 2]) if n > 2 else 1]
    vecs = vecs[:, :n] * np.array(signs[:n])

    names = ["Level", "Slope", "Curvature", *[f"PC{i}" for i in range(4, n + 1)]][:n]
    loadings = pd.DataFrame(vecs, index=dx.columns, columns=names)
    scores = pd.DataFrame((dx - dx.mean()).values @ vecs, index=dx.index, columns=names)
    explained = pd.Series(vals[:n] / vals.sum(), index=names)
    return explained, loadings, scores


def pca_shock(loadings, pc, size_bp):
    """Turn a PC loading into a curve shock function: size_bp * normalised loading."""
    tenors = loadings.index.to_numpy(float)
    v = loadings[pc].to_numpy()
    v = v / np.abs(v).max()
    return lambda t: 1e-4 * size_bp * np.interp(np.asarray(t, float), tenors, v)
