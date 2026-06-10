"""Slot-limited orchestrator utilization: predictions vs the truth.

An orchestrator with ``K`` execution slots can hold at most ``K`` simultaneous
positions. The number of signals active at a step, ``A_t = sum_i 1[active_{i,t}]``,
is a sum of *correlated* Bernoullis (from the latent model). We measure:

* ``utilization = E[min(A, K)] / K`` -- fraction of slot-capacity used,
* ``fill_rate`` (single-slot ``P(A >= 1)``) -- the post's headline quantity,
* ``overflow`` -- ``P(A > K)`` -- demand the slots cannot absorb.

We compare four *predictions* of ``P(>=1 active)`` / utilization, all evaluated
against the simulated truth:

1. **naive-independent (N)** -- treat the ``N`` signals as independent:
   ``P(>=1) = 1 - (1-p)^N``. Ignores correlation entirely.
2. **post-heuristic, latent-fed** -- the blog post's ``1 - (1-p)^{N_eff}`` with a
   *non-integer* ``N_eff`` computed from the mean latent correlation. The latent
   correlation is the larger of the two candidate inputs, so this variant has the
   smallest ``N_eff`` and the deepest undershoot of P(>=1).
3. **post-heuristic, binary-fed** -- the same formula fed the mean Pearson
   correlation of the binary activations (what a practitioner measuring the
   signals themselves would use). Tetrachoric attenuation makes the binary
   correlation smaller, hence ``N_eff`` larger and the prediction closer to the
   naive independent one -- feeding the binary correlation *shrinks* the
   heuristic's undershoot.
4. **correlated-Bernoulli (truth)** -- the empirical distribution of ``A_t`` from
   the correlated latent model, which correctly captures clustering.

Which correlation is fed changes the size of the error and whether the heuristic
beats the naive model at all -- one of the paper's quantitative results.
"""

from __future__ import annotations

import numpy as np

from .breadth import binary_corr_matrix, equicorr_neff, mean_offdiag


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
# the P(>=1 active) predictions
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
    """Naive + BOTH post-heuristic feeding variants vs the simulated truth.

    The heuristic needs a scalar ``rho_bar``; which correlation is fed changes
    the answer, so we evaluate both:

    * ``_latent`` -- the mean off-diagonal population latent correlation
      (unobservable in practice). It is the larger input, so ``N_eff`` is
      smallest and the predicted P(>=1) undershoots the most.
    * ``_binary`` -- the mean off-diagonal Pearson correlation of the realized
      binary activations (the observable input a practitioner would measure).
      Tetrachoric attenuation makes it smaller, so ``N_eff`` is larger and the
      prediction sits closer to the naive independent one, shrinking the
      undershoot.
    """
    cfg = sim.cfg
    n, p = cfg.n_pairs, cfg.p_active
    rho_latent = mean_offdiag(sim.corr_latent)
    rho_binary = mean_offdiag(binary_corr_matrix(sim.activations))

    sim_u = simulated_utilization(sim.activations, k_slots)
    truth = sim_u["p_at_least_one"]

    naive = pred_naive_independent(n, p)
    heur_latent = pred_post_heuristic(n, p, rho_latent)
    heur_binary = pred_post_heuristic(n, p, rho_binary)

    neff_latent = equicorr_neff(n, rho_latent)
    neff_binary = equicorr_neff(n, rho_binary)

    return {
        "k_slots": int(k_slots),
        "rho_latent": rho_latent,
        "rho_binary": rho_binary,
        "p_at_least_one_sim": truth,
        "p_at_least_one_naive": naive,
        "p_at_least_one_heuristic_latent": heur_latent,
        "p_at_least_one_heuristic_binary": heur_binary,
        "err_naive": naive - truth,
        "err_heuristic_latent": heur_latent - truth,
        "err_heuristic_binary": heur_binary - truth,
        "abs_err_naive": abs(naive - truth),
        "abs_err_heuristic_latent": abs(heur_latent - truth),
        "abs_err_heuristic_binary": abs(heur_binary - truth),
        "utilization_sim": sim_u["utilization"],
        "fill_rate_sim": sim_u["fill_rate"],
        "overflow_sim": sim_u["overflow"],
        "mean_active_sim": sim_u["mean_active"],
        # post's analytic mean active = N_eff * p, under each feeding choice
        "mean_active_post_latent": (neff_latent * p)
        if np.isfinite(neff_latent) else float("nan"),
        "mean_active_post_binary": (neff_binary * p)
        if np.isfinite(neff_binary) else float("nan"),
        # truth for mean active is just N * p regardless of correlation (linearity)
        "mean_active_independent": n * p,
    }
