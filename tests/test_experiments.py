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
    heuristic underestimates it (clustering reduces the union)."""
    cfg = SignalConfig(n_pairs=15, p_active=0.1, rho=0.4, structure="equicorr", t_steps=120_000)
    sim = simulate(cfg, np.random.default_rng(5))
    u = simulated_utilization(sim.activations, k_slots=1)
    naive = pred_naive_independent(cfg.n_pairs, cfg.p_active)
    rho_lat = mean_offdiag(sim.corr_latent)
    heur = pred_post_heuristic(cfg.n_pairs, cfg.p_active, rho_lat)
    assert naive > u["p_at_least_one"] > 0
    assert heur < u["p_at_least_one"]  # post heuristic undershoots the simulated truth


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
              "realized_neff_pnl", "rho_latent", "rho_binary",
              "relerr_neff_post_binary", "util_p_at_least_one_sim_k1"):
        assert k in r and (np.isfinite(r[k]) or isinstance(r[k], (int, float)))
