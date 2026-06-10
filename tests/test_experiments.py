"""Fast sanity tests for the signal-correlation simulation and estimators.

Run: python -m pytest -q   (from the project root)
"""

from __future__ import annotations

import numpy as np
import pytest

from signal_experiments import (
    SignalConfig,
    binary_corr_matrix,
    canonical_configs,
    equicorr_neff,
    equicorr_psd_floor,
    is_psd_valid_equicorr,
    meucci_enb,
    phi_from_tetrachoric,
    realized_effective_n,
    run_experiment,
    simulate,
)
from signal_experiments.breadth import mean_offdiag
from signal_experiments.utilization import (
    pred_naive_independent,
    pred_post_heuristic,
    simulated_utilization,
    utilization_predictions,
)


def test_activation_prob_matches_p():
    """Simulated marginal activation probability ~ p."""
    cfg = SignalConfig(n_pairs=10, p_active=0.12, rho=0.3, structure="equicorr", t_steps=80_000)
    sim = simulate(cfg, np.random.default_rng(0))
    assert abs(sim.activations.mean() - cfg.p_active) < 0.01


def test_latent_correlation_recovered():
    """Empirical latent correlation ~ configured rho for equicorrelation."""
    cfg = SignalConfig(n_pairs=8, p_active=0.1, rho=0.4, structure="equicorr", t_steps=80_000)
    sim = simulate(cfg, np.random.default_rng(1))
    emp = mean_offdiag(np.corrcoef(sim.latent.T))
    assert abs(emp - 0.4) < 0.03


def test_binary_corr_below_latent():
    """Binary-signal correlation is strictly attenuated relative to latent rho."""
    cfg = SignalConfig(n_pairs=10, p_active=0.1, rho=0.5, structure="equicorr", t_steps=120_000)
    sim = simulate(cfg, np.random.default_rng(2))
    rho_bin = mean_offdiag(binary_corr_matrix(sim.activations))
    assert rho_bin < cfg.rho - 0.05  # meaningfully below the latent correlation


def test_tetrachoric_prediction_accurate():
    """Population tetrachoric phi predicts the realized binary correlation."""
    cfg = SignalConfig(n_pairs=12, p_active=0.1, rho=0.45, structure="equicorr", t_steps=200_000)
    sim = simulate(cfg, np.random.default_rng(3))
    rho_bin = mean_offdiag(binary_corr_matrix(sim.activations))
    phi = phi_from_tetrachoric(cfg.rho, cfg.p_active)
    assert abs(phi - rho_bin) < 0.02


def test_realized_n_equals_equicorr_in_homogeneous_signal_case():
    """Variance reduction of the binary signal streams equals the equicorrelation
    N_eff fed the binary correlation -- the homogeneous identity the post uses."""
    cfg = SignalConfig(n_pairs=10, p_active=0.1, rho=0.3, structure="equicorr", t_steps=150_000)
    sim = simulate(cfg, np.random.default_rng(4))
    realized = realized_effective_n(sim.activations.astype(float))
    rho_bin = mean_offdiag(binary_corr_matrix(sim.activations))
    neff = equicorr_neff(cfg.n_pairs, rho_bin)
    assert abs(realized - neff) / neff < 0.05


def test_psd_guard():
    """N_eff returns nan outside the PSD-valid equicorrelation range."""
    n = 10
    floor = equicorr_psd_floor(n)
    assert floor == pytest.approx(-1 / 9)
    assert not is_psd_valid_equicorr(n, floor - 0.05)
    assert np.isnan(equicorr_neff(n, -0.5))           # below PSD floor
    assert np.isfinite(equicorr_neff(n, 0.0))


def test_psd_invalid_config_raises():
    cfg = SignalConfig(n_pairs=5, p_active=0.1, rho=-0.5, structure="equicorr", t_steps=1000)
    with pytest.raises(ValueError):
        simulate(cfg, np.random.default_rng(0))


def test_meucci_enb_bounds():
    """ENB is N for identity and ~1 for a single-factor (rank-1-ish) matrix."""
    n = 8
    assert meucci_enb(np.eye(n)) == pytest.approx(n, rel=1e-6)
    ones = 0.999 * np.ones((n, n)) + 0.001 * np.eye(n)
    assert meucci_enb(ones) < 1.2


def test_utilization_predictions_order():
    """Naive (ignores corr) overestimates P(>=1) vs correlated truth; the post's
    latent-fed heuristic underestimates it (clustering reduces the union)."""
    cfg = SignalConfig(n_pairs=15, p_active=0.1, rho=0.4, structure="equicorr", t_steps=120_000)
    sim = simulate(cfg, np.random.default_rng(5))
    u = simulated_utilization(sim.activations, k_slots=1)
    naive = pred_naive_independent(cfg.n_pairs, cfg.p_active)
    rho_lat = mean_offdiag(sim.corr_latent)
    heur = pred_post_heuristic(cfg.n_pairs, cfg.p_active, rho_lat)
    assert naive > u["p_at_least_one"] > 0
    assert heur < u["p_at_least_one"]  # post heuristic undershoots the simulated truth


def test_capacity_variants_present_and_ordered():
    """utilization_predictions exposes BOTH heuristic feedings; the binary corr
    is tetrachorically attenuated -> larger N_eff -> larger P(>=1) prediction,
    sitting between the latent-fed prediction and the naive one."""
    cfg = SignalConfig(n_pairs=12, p_active=0.10, rho=0.5, structure="equicorr", t_steps=60_000)
    sim = simulate(cfg, np.random.default_rng(9))
    u = utilization_predictions(sim, k_slots=1)
    for key in ("p_at_least_one_heuristic_latent", "p_at_least_one_heuristic_binary",
                "abs_err_heuristic_latent", "abs_err_heuristic_binary",
                "err_heuristic_latent", "err_heuristic_binary",
                "mean_active_post_latent", "mean_active_post_binary"):
        assert key in u and np.isfinite(u[key]), key
    assert u["p_at_least_one_heuristic_binary"] > u["p_at_least_one_heuristic_latent"]
    assert u["p_at_least_one_heuristic_binary"] <= u["p_at_least_one_naive"] + 1e-9
    assert u["mean_active_post_binary"] > u["mean_active_post_latent"]


def test_beta_is_sampled_from_grid():
    """factor_return_beta is an explicit sampled batch parameter (not a hidden
    constant): sample_config draws every value of SAMPLED_BETA_GRID, and the
    value is recorded in the experiment record."""
    from signal_experiments.model import SAMPLED_BETA_GRID, sample_config

    rng = np.random.default_rng(0)
    betas = {sample_config(rng, t_steps=1000).factor_return_beta for _ in range(300)}
    assert betas == set(SAMPLED_BETA_GRID)
    cfg = sample_config(np.random.default_rng(1), t_steps=2000)
    rec = run_experiment(cfg, np.random.default_rng(2))
    assert rec["cfg_factor_return_beta"] == cfg.factor_return_beta


def test_het_loadings_unit_variance_worst_case():
    """Worst-case heterogeneous config (high rho + max block share + max
    dispersion): loading rows are renormalized so total latent variance is
    exactly 1 and the empirical activation probability matches the configured
    p (the old code silently clipped idio variance and violated p)."""
    from signal_experiments.model import latent_loadings

    cfg = SignalConfig(n_pairs=12, p_active=0.10, rho=0.65, structure="factor",
                       n_blocks=3, block_share=0.25, loading_dispersion=0.45,
                       t_steps=200_000)
    b, idio_sd = latent_loadings(cfg, np.random.default_rng(7))
    total_var = np.sum(b**2, axis=1) + idio_sd**2
    assert np.allclose(total_var, 1.0, atol=1e-12)

    sim = simulate(cfg, np.random.default_rng(8))
    assert abs(sim.activations.mean() - cfg.p_active) < 0.01
    assert np.all(np.abs(sim.latent.var(axis=0) - 1.0) < 0.03)


def test_optimal_pairs_het_config_is_heterogeneous():
    """The heterogeneous optimal-N run must use genuinely dispersed loadings
    (the old wiring dropped dispersion/blocks and degenerated to equicorr)."""
    from signal_experiments.model import latent_corr

    cfg = SignalConfig(n_pairs=12, p_active=0.15, rho=0.30, structure="factor",
                       n_blocks=3, block_share=0.10, loading_dispersion=0.35,
                       t_steps=1000)
    corr = latent_corr(cfg, np.random.default_rng(11))
    iu = np.triu_indices(12, k=1)
    assert np.std(corr[iu]) > 0.05  # genuinely unequal pairwise correlations


def test_optimal_pairs_reports_both_variants():
    from signal_experiments import optimal_pairs

    res = optimal_pairs(p_active=0.2, rho=0.3, structure="factor", n_blocks=2,
                        loading_dispersion=0.3, block_share=0.1, max_pairs=4,
                        t_steps=4000, reps=2, seed=1)
    assert res["params"]["loading_dispersion"] == 0.3
    for key in ("optimal_n_post_latent", "optimal_n_post_binary", "optimal_n_sim",
                "gap_latent", "gap_binary",
                "throughput_cost_pct_latent", "throughput_cost_pct_binary"):
        assert key in res, key


def test_mean_active_is_n_times_p():
    """E[# active] = N*p regardless of correlation (linearity of expectation),
    contradicting the post's N_eff*p point estimate."""
    cfg = SignalConfig(n_pairs=20, p_active=0.1, rho=0.6, structure="equicorr", t_steps=120_000)
    sim = simulate(cfg, np.random.default_rng(6))
    mean_active = sim.activations.sum(axis=1).mean()
    assert abs(mean_active - cfg.n_pairs * cfg.p_active) < 0.1


def test_run_experiment_is_deterministic():
    cfg = canonical_configs(t_steps=8000)["heterogeneous"]
    a = run_experiment(cfg, np.random.default_rng(123))
    b = run_experiment(cfg, np.random.default_rng(123))
    assert a["realized_neff_pnl"] == b["realized_neff_pnl"]
    assert a["rho_binary"] == b["rho_binary"]


def test_record_has_expected_keys():
    cfg = canonical_configs(t_steps=8000)["homogeneous"]
    r = run_experiment(cfg, np.random.default_rng(0))
    for k in ("neff_post_binary", "neff_post_latent", "enb_meucci_latent",
              "realized_neff_pnl", "realized_neff_signal",
              "rho_latent", "rho_binary", "cfg_factor_return_beta",
              "relerr_neff_post_binary", "relerr_neff_post_binary_vs_signal",
              "util_p_at_least_one_sim_k1",
              "util_abs_err_heuristic_latent_k1",
              "util_abs_err_heuristic_binary_k1",
              "util_mean_active_post_latent_k3",
              "util_mean_active_post_binary_k3"):
        assert k in r and (np.isfinite(r[k]) or isinstance(r[k], (int, float)))
