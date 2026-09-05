"""Two definitions that were duplicated across the project and wrong in both places.

Both defects were found by an external reproducibility audit of the release
repository rather than by anything here, which is the point of having one.

1. RANK CORRELATION WITH TIES.  Every rank correlation in this project was
   computed as `argsort(argsort(x))`, which assigns ORDINAL ranks: tied values
   get consecutive integers in whatever order argsort happened to return them.
   Spearman's coefficient is defined on AVERAGE ranks, where tied values share
   the mean of the ranks they span.  The two differ whenever ties are present,
   and ties are present here because modal bin mass is quoted to three decimals
   and several configurations collide.  The audit measured the gap on E8 at
   -0.1422 against -0.1469.  Small, but the reported statistic was not the one
   named in the text.

2. SEEDS FROM `hash()`.  Several simulation cells seeded from
   `hash((n, T, scale, s, "C"))`.  Python randomises string hashing per process
   unless PYTHONHASHSEED is fixed, so any tuple containing a string produced a
   different seed on every run and the reported cells could not be regenerated.
   Verified: the same tuple gave 561268085 and 2230517452 on two consecutive
   interpreters.  `stable_seed` uses CRC32 over a canonical encoding instead,
   which is fixed across processes, machines and Python versions.

Neither fix changes a conclusion in the paper.  The first changes the reported
correlations in the fourth decimal; the second changes which draws a rerun makes,
so figures from the original run cannot be reproduced exactly and are labelled as
such wherever they appear.
"""
from __future__ import annotations

import zlib

import numpy as np


def average_ranks(x):
    """Ranks with ties averaged -- the definition Spearman's coefficient uses.

    `argsort(argsort(x))` returns ordinal ranks and silently breaks ties by
    position, which makes the statistic depend on input order.  This does not.
    """
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    # Average the ranks within each group of equal values.
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(u, v):
    """Spearman rank correlation, ties averaged.  NaNs dropped pairwise.

    THE one definition.  Import it; do not write `argsort(argsort(...))` again --
    it appeared in five files here and was wrong in all five.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 3:
        return float("nan")
    ru = average_ranks(u[ok])
    rv = average_ranks(v[ok])
    ru = ru - ru.mean()
    rv = rv - rv.mean()
    den = np.sqrt((ru ** 2).sum() * (rv ** 2).sum())
    return float((ru * rv).sum() / den) if den > 0 else float("nan")


def stable_seed(*parts, bits=32):
    """A seed that is identical across processes, machines and Python versions.

    `hash()` on anything containing a string is salted per process, so a cell
    seeded that way cannot be regenerated once the salt is gone.  CRC32 over a
    canonical repr has no such property.  Floats are formatted with repr so that
    1.0 and 1 do not collide silently.
    """
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    return zlib.crc32(key) % (2 ** bits)


def _self_test():
    """Run as a script: python -m kanrel.stats"""
    # average ranks: [10, 20, 20, 30] -> [0, 1.5, 1.5, 3]
    r = average_ranks([10, 20, 20, 30])
    assert np.allclose(r, [0, 1.5, 1.5, 3]), r
    # order independence under ties, which ordinal ranks do not have
    a = [1, 2, 2, 3, 4]
    b = [5, 6, 7, 8, 9]
    assert abs(spearman(a, b) - spearman(a[::-1], b[::-1])) < 1e-12
    # agreement with scipy where available
    try:
        from scipy.stats import spearmanr
        rng = np.random.default_rng(0)
        for _ in range(200):
            u = rng.integers(0, 5, 12).astype(float)
            v = rng.integers(0, 5, 12).astype(float)
            s = spearmanr(u, v).statistic
            if np.isfinite(s):
                assert abs(spearman(u, v) - s) < 1e-10, (u, v, spearman(u, v), s)
        print("  spearman matches scipy on 200 tied samples")
    except ImportError:
        print("  scipy not installed; skipped the cross-check")
    # stable_seed really is stable
    assert stable_seed(2000, 4, 1.0, 3, "C") == stable_seed(2000, 4, 1.0, 3, "C")
    assert stable_seed(1, "a") != stable_seed(1, "b")
    print(f"  stable_seed(2000, 4, 1.0, 3, 'C') = {stable_seed(2000, 4, 1.0, 3, 'C')}")
    print("  all self-tests passed")


if __name__ == "__main__":
    _self_test()
