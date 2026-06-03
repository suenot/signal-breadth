"""Slot-limited orchestrator utilization: three predictions vs the truth.

An orchestrator with ``K`` execution slots can hold at most ``K`` simultaneous
positions. The number of signals active at a step, ``A_t = sum_i 1[active_{i,t}]``,
is a sum of *correlated* Bernoullis (from the latent model). We measure:

* ``utilization = E[min(A, K)] / K`` -- fraction of slot-capacity used,
* ``fill_rate`` (single-slot ``P(A >= 1)``) -- the post's headline quantity,
* ``overflow`` -- ``P(A > K)`` -- demand the slots cannot absorb.

We compare three *predictions* of ``P(>=1 active)`` / utilization, all evaluated
against the simulated truth:

1. **naive-independent (N)** -- treat the ``N`` signals as independent:
   ``P(>=1) = 1 - (1-p)^N``. Ignores correlation entirely.
2. **post-heuristic (N_eff)** -- the blog post's ``1 - (1-p)^{N_eff}`` with a
   *non-integer* ``N_eff`` plugged into the independent-Bernoulli formula.
3. **correlated-Bernoulli (truth)** -- the empirical distribution of ``A_t`` from
   the correlated latent model, which correctly captures clustering.

The systematic error of (2) vs (3) -- which we show *grows with rho* -- is one of
the paper's quantitative results.
"""

from __future__ import annotations

import numpy as np

from .breadth import equicorr_neff, mean_offdiag


# --------------------------------------------------------------------------- #
# ground-truth utilization from the simulated activation matrix
# --------------------------------------------------------------------------- #
def active_counts(activations: np.ndarray) -> np.ndarray:
    """Number of simultaneously-active signals per step, shape (T,)."""
    return activations.sum(axis=1)


def simulated_utilization(activations: np.ndarray, k_slots: int) -> dict:
    """Empirical utilization / fill / overflow for ``k_slots`` execution slots."""
    a = active_counts(activations)
    capped = np.minimum(a, k_slots)
    return {
        "k_slots": int(k_slots),
        "p_at_least_one": float(np.mean(a >= 1)),
        "utilization": float(np.mean(capped) / k_slots),
        "fill_rate": float(capped.sum() / max(a.sum(), 1)),  # served / demanded
        "overflow": float(np.mean(a > k_slots)),
        "mean_active": float(a.mean()),
        "var_active": float(a.var()),
    }


# --------------------------------------------------------------------------- #
# the three P(>=1 active) predictions
# --------------------------------------------------------------------------- #
def pred_naive_independent(n: int, p: float) -> float:
    """Naive: signals independent -> ``1 - (1-p)^N``."""
    return 1.0 - (1.0 - p) ** n


def pred_post_heuristic(n: int, p: float, rho_bar: float) -> float:
    """The blog post's ``1 - (1-p)^{N_eff}`` with ``N_eff`` from equicorrelation.

    Plugging a non-integer ``N_eff`` into the independent-Bernoulli formula is the
    unjustified heuristic the paper scrutinizes.
    """
    neff = equicorr_neff(n, rho_bar)
    if not np.isfinite(neff):
        return float("nan")
    return 1.0 - (1.0 - p) ** neff


def utilization_predictions(sim, k_slots: int = 1) -> dict:
    """All three P(>=1) predictions plus the simulated truth, and their errors.

    ``rho_bar`` for the heuristic is taken as the mean off-diagonal *latent*
    correlation (the most charitable input; using binary correlation only shrinks
    N_eff further and worsens the gap, which we also report in analysis).
    """
    cfg = sim.cfg
    n, p = cfg.n_pairs, cfg.p_active
    rho_latent = mean_offdiag(sim.corr_latent)

    sim_u = simulated_utilization(sim.activations, k_slots)
    truth = sim_u["p_at_least_one"]

    naive = pred_naive_independent(n, p)
    heuristic = pred_post_heuristic(n, p, rho_latent)

    return {
        "k_slots": int(k_slots),
        "rho_latent": rho_latent,
        "p_at_least_one_sim": truth,
        "p_at_least_one_naive": naive,
        "p_at_least_one_heuristic": heuristic,
        "err_naive": naive - truth,
        "err_heuristic": heuristic - truth,
        "abs_err_naive": abs(naive - truth),
        "abs_err_heuristic": abs(heuristic - truth),
        "utilization_sim": sim_u["utilization"],
        "fill_rate_sim": sim_u["fill_rate"],
        "overflow_sim": sim_u["overflow"],
        "mean_active_sim": sim_u["mean_active"],
        # post's analytic mean active = N_eff * p (Jensen-style point estimate)
        "mean_active_post": (equicorr_neff(n, rho_latent) * p)
        if np.isfinite(equicorr_neff(n, rho_latent)) else float("nan"),
        # truth for mean active is just N * p regardless of correlation (linearity)
        "mean_active_independent": n * p,
    }
