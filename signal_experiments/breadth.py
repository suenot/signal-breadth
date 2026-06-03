"""Effective-breadth estimators and the binary-vs-latent correlation relation.

Four quantities, all operating on the simulated data:

* ``equicorr_neff`` -- the blog post's ``N / (1 + (N-1) rho_bar)``, the
  equicorrelation effective-sample-size formula. Guarded to the PSD-valid range.
* ``meucci_enb`` -- Meucci's Effective Number of Bets: the exponential of the
  entropy of the PCA variance distribution of a correlation matrix
  (Meucci, *Managing Diversification*, Risk 2009). A principled, structure-aware
  baseline that does *not* assume equicorrelation.
* ``realized_effective_n`` -- the ground-truth effective breadth: the variance
  reduction of the equal-weight portfolio relative to a single strategy,
  ``Var(single) / Var(equal-weight portfolio)``. For an equicorrelation world
  this equals the equicorr ``N_eff`` in expectation; in general it is the honest
  target both formulas are approximating.
* the *tetrachoric* relation -- thresholding correlated Gaussians attenuates
  correlation: the Pearson correlation of the binary activations is strictly
  smaller (in magnitude) than the latent Gaussian correlation that produced
  them. So feeding *signal* correlation into ``N_eff`` (as the post does) is not
  the same as feeding *latent/return* correlation; the two give different breadth.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# the blog post's formula
# --------------------------------------------------------------------------- #
def equicorr_neff(n: int, rho_bar: float) -> float:
    """Equicorrelation effective number of pairs ``N / (1 + (N-1) rho_bar)``.

    This is exactly the source article's formula. It is only a valid effective
    sample size when the underlying equicorrelation matrix is PSD, i.e.
    ``rho_bar >= -1/(N-1)``; outside that the denominator is meaningless. We clip
    to the valid range and return ``nan`` if asked outside it so callers can flag
    the misuse rather than silently returning a number.
    """
    if n <= 1:
        return float(n)
    floor = -1.0 / (n - 1)
    if rho_bar < floor - 1e-9 or rho_bar > 1.0 + 1e-9:
        return float("nan")
    denom = 1.0 + (n - 1) * rho_bar
    if denom <= 1e-12:
        return float("nan")
    return n / denom


def correlation_factor(n: int, rho_bar: float) -> float:
    """The post's ``C_f = 1 + (N-1) rho_bar`` (so ``N_eff = N / C_f``)."""
    return 1.0 + (n - 1) * rho_bar


# --------------------------------------------------------------------------- #
# Meucci Effective Number of Bets (principled baseline)
# --------------------------------------------------------------------------- #
def meucci_enb(corr: np.ndarray) -> float:
    """Effective Number of Bets via the entropy of the PCA variance spectrum.

    ``ENB = exp(-sum_k w_k ln w_k)`` where ``w_k`` are the eigenvalue shares of
    the correlation matrix (each principal component is an uncorrelated "bet";
    ``w_k`` is its share of total variance). ENB ranges from 1 (one dominant
    factor) to ``N`` (isotropic / independent). Unlike the equicorrelation
    formula it uses the *whole* correlation structure, so it stays accurate under
    heterogeneous/block correlations.
    """
    evals = np.linalg.eigvalsh(corr)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    if total <= 0:
        return float("nan")
    w = evals / total
    w = w[w > 1e-15]
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))


# --------------------------------------------------------------------------- #
# ground-truth realized effective breadth
# --------------------------------------------------------------------------- #
def realized_effective_n(returns: np.ndarray) -> float:
    """Variance-reduction effective N of the equal-weight portfolio.

    ``= mean_i Var(r_i) / Var( mean_i r_i )``. For ``N`` series with average
    pairwise correlation ``rho`` and equal variance this equals
    ``N / (1 + (N-1) rho)`` -- i.e. the equicorrelation formula is *exactly* the
    variance-reduction ratio when correlations are homogeneous. With unequal
    variances/correlations it is the honest generalization and is what the scalar
    formulas approximate. Uses per-series returns (here, the signed PnL stream).
    """
    n = returns.shape[1]
    var_single = np.mean(np.var(returns, axis=0, ddof=1))
    port = returns.mean(axis=1)
    var_port = np.var(port, ddof=1)
    if var_port <= 1e-18:
        return float("nan")
    return float(var_single / var_port)


def realized_effective_n_from_corr(corr: np.ndarray) -> float:
    """Variance-reduction effective N assuming equal variances, from a
    correlation matrix: ``N^2 / sum_{ij} corr_{ij}`` (= ``N / (1 + (N-1) mean_offdiag)``
    for equicorrelation). Lets us compute the target from either latent or binary
    correlation and compare what each implies."""
    n = corr.shape[0]
    total = corr.sum()
    if total <= 1e-12:
        return float("nan")
    return float(n * n / total)


# --------------------------------------------------------------------------- #
# binary vs latent correlation (tetrachoric attenuation)
# --------------------------------------------------------------------------- #
def mean_offdiag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(np.mean(corr[iu]))


def binary_corr_matrix(activations: np.ndarray) -> np.ndarray:
    """Pearson (phi) correlation matrix of the 0/1 activation columns."""
    x = activations.astype(float)
    # guard constant columns (no activations) -> corrcoef gives nan; set to identity row
    sd = x.std(axis=0)
    corr = np.corrcoef(x.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    bad = sd <= 1e-12
    if bad.any():  # constant column: uncorrelated with everything by convention
        corr[bad, :] = 0.0
        corr[:, bad] = 0.0
        corr[bad, bad] = 1.0
    return corr


def phi_from_tetrachoric(rho: float, p: float) -> float:
    """Population Pearson (phi) correlation of two binary variables obtained by
    thresholding a bivariate normal with correlation ``rho`` at the ``1-p``
    quantile (equal marginals ``p``).

    ``phi = (P(both active) - p^2) / (p (1-p))`` where ``P(both active)`` is the
    bivariate-normal upper-orthant probability. This is the inverse of the
    tetrachoric correlation and is always smaller in magnitude than ``rho`` for
    ``0 < p < 1`` -- the "binary attenuation" that makes signal correlation a
    biased proxy for latent correlation.
    """
    if p <= 0 or p >= 1:
        return 0.0
    thr = stats.norm.ppf(1.0 - p)
    # P(X>thr, Y>thr) for standard bivariate normal corr rho
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    p_both = float(mvn.cdf([-thr, -thr]))  # P(X<=-thr, Y<=-thr) = P(X>thr,Y>thr) by symmetry
    return (p_both - p * p) / (p * (1.0 - p))


def breadth_bundle(sim) -> dict:
    """Compute every breadth quantity for one simulated portfolio (``sim`` is a
    ``model.SimulatedSignals``)."""
    cfg = sim.cfg
    n = cfg.n_pairs

    corr_latent = sim.corr_latent
    corr_binary = binary_corr_matrix(sim.activations)
    rho_latent = mean_offdiag(corr_latent)
    rho_binary = mean_offdiag(corr_binary)

    # realized binary activation prob (sanity: ~ p)
    p_hat = float(sim.activations.mean())

    # the post's N_eff fed two ways: latent-corr (idealized) and binary-signal-corr
    neff_latent = equicorr_neff(n, rho_latent)
    neff_binary = equicorr_neff(n, rho_binary)

    # realized PnL (return) correlation matrix -- the structure that actually
    # determines portfolio diversification. ENB on THIS matrix is the principled,
    # structure-aware competitor to the equicorrelation N_eff.
    corr_returns = np.corrcoef(sim.returns.T)
    corr_returns = np.nan_to_num(corr_returns, nan=0.0)
    np.fill_diagonal(corr_returns, 1.0)
    rho_returns = mean_offdiag(corr_returns)

    enb_latent = meucci_enb(corr_latent)
    enb_binary = meucci_enb(corr_binary)
    enb_returns = meucci_enb(corr_returns)

    # GROUND TRUTH: variance reduction of the equal-weight realized-PnL portfolio.
    # This is the honest "effective number of independent bets" -- the actual
    # Sharpe benefit of running N correlated strategies. NOT algebraically tied to
    # any single mean correlation, so it is a fair target for all estimators.
    realized_pnl = realized_effective_n(sim.returns)
    # secondary target: variance reduction of the binary signal streams themselves
    realized_signal = realized_effective_n(sim.activations.astype(float))

    # tetrachoric prediction of the binary correlation (population)
    phi_pred = phi_from_tetrachoric(rho_latent, cfg.p_active)

    return {
        "n_pairs": n,
        "p_active": cfg.p_active,
        "p_hat": p_hat,
        "rho_cfg": cfg.rho,
        "rho_latent": rho_latent,
        "rho_binary": rho_binary,
        "rho_returns": rho_returns,
        "phi_pred_tetrachoric": phi_pred,
        "binary_attenuation": rho_binary / rho_latent if abs(rho_latent) > 1e-9 else float("nan"),
        "neff_post_latent": neff_latent,
        "neff_post_binary": neff_binary,
        "enb_meucci_latent": enb_latent,
        "enb_meucci_binary": enb_binary,
        "enb_meucci_returns": enb_returns,
        "realized_neff_pnl": realized_pnl,
        "realized_neff_signal": realized_signal,
    }
