"""Controlled validation of effective-breadth approximations for correlated
trading signals (the ``N_eff = N/(1+(N-1)rho)`` heuristic and its utilization
companion ``1-(1-p)^{N_eff}``), benchmarked against Meucci's Effective Number of
Bets and the correct correlated-Bernoulli model."""

from .breadth import (
    binary_corr_matrix,
    breadth_bundle,
    correlation_factor,
    equicorr_neff,
    meucci_enb,
    phi_from_tetrachoric,
    realized_effective_n,
    realized_effective_n_from_corr,
)
from .simulate import optimal_pairs, rho_sweep, run_batch, run_experiment
from .model import (
    SignalConfig,
    SimulatedSignals,
    canonical_configs,
    equicorr_psd_floor,
    is_psd_valid_equicorr,
    latent_corr,
    sample_config,
    simulate,
)
from .utilization import (
    pred_naive_independent,
    pred_post_heuristic,
    simulated_utilization,
    utilization_predictions,
)

__all__ = [
    # model
    "SignalConfig",
    "SimulatedSignals",
    "simulate",
    "sample_config",
    "canonical_configs",
    "latent_corr",
    "equicorr_psd_floor",
    "is_psd_valid_equicorr",
    # breadth
    "equicorr_neff",
    "correlation_factor",
    "meucci_enb",
    "realized_effective_n",
    "realized_effective_n_from_corr",
    "binary_corr_matrix",
    "phi_from_tetrachoric",
    "breadth_bundle",
    # utilization
    "pred_naive_independent",
    "pred_post_heuristic",
    "simulated_utilization",
    "utilization_predictions",
    # experiments
    "run_experiment",
    "run_batch",
    "rho_sweep",
    "optimal_pairs",
]
__version__ = "0.1.0"
