"""One experiment = one simulated correlated-signal portfolio, end to end.

We draw a ground-truth config (homogeneous or heterogeneous correlation),
simulate the (T, N) activation + return matrices, and record every breadth
estimator, the binary-vs-latent correlation gap, and the slot-utilization
predictions vs the simulated truth. Batches of these records are the raw
material for the paper's analysis.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .breadth import breadth_bundle
from .model import SignalConfig, sample_config, simulate
from .utilization import utilization_predictions


def run_experiment(
    cfg: SignalConfig,
    rng: np.random.Generator,
    *,
    k_slots_list: tuple[int, ...] = (1, 3),
) -> dict:
    """Simulate one portfolio and return a flat record (config + measures)."""
    sim = simulate(cfg, rng)
    b = breadth_bundle(sim)

    # GROUND TRUTH: realized PnL variance reduction (the true effective breadth /
    # diversification benefit). Every estimator is scored against this.
    truth = b["realized_neff_pnl"]
    # secondary target: variance reduction of the binary signal streams
    truth_signal = b["realized_neff_signal"]

    def err(pred: float, tgt: float) -> float:
        if not (np.isfinite(pred) and np.isfinite(tgt)) or tgt <= 1e-9:
            return float("nan")
        return (pred - tgt) / tgt  # signed relative error

    record: dict = {
        **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
        **b,
        # breadth accuracy vs the realized PnL effective N (the honest target):
        #   - neff_post_binary  = exactly what the blog post computes (binary corr)
        #   - neff_post_latent  = the formula fed the (unobservable) latent corr
        #   - enb_meucci_latent = principled PCA baseline on the population latent
        #     correlation (the fairest possible input for ENB)
        "relerr_neff_post_binary": err(b["neff_post_binary"], truth),
        "relerr_neff_post_latent": err(b["neff_post_latent"], truth),
        "relerr_enb_latent": err(b["enb_meucci_latent"], truth),
        "relerr_enb_returns": err(b["enb_meucci_returns"], truth),
        # accuracy against the signal-stream effective N (a second target)
        "relerr_neff_post_binary_vs_signal": err(b["neff_post_binary"], truth_signal),
        "relerr_enb_latent_vs_signal": err(b["enb_meucci_latent"], truth_signal),
        # binary attenuation gap (latent - binary mean correlation)
        "corr_gap_latent_minus_binary": b["rho_latent"] - b["rho_binary"],
        "is_homogeneous": int(cfg.structure == "equicorr"),
    }

    # utilization predictions for each slot count
    for k in k_slots_list:
        u = utilization_predictions(sim, k_slots=k)
        suffix = f"_k{k}"
        for key, val in u.items():
            if key == "k_slots":
                continue
            record[f"util_{key}{suffix}"] = val
    return record


def run_batch(
    n_experiments: int,
    *,
    structure: str = "random",
    seed: int = 0,
    t_steps: int = 20_000,
    k_slots_list: tuple[int, ...] = (1, 3),
    progress_every: int = 0,
) -> list[dict]:
    """Run ``n_experiments`` with independently-seeded child RNGs (reproducible)."""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_experiments)
    records = []
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        cfg = sample_config(rng, structure=structure, t_steps=t_steps)
        records.append(run_experiment(cfg, rng, k_slots_list=k_slots_list))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_experiments}", flush=True)
    return records


# --------------------------------------------------------------------------- #
# rho sweep for the utilization-heuristic-error figure (homogeneous world)
# --------------------------------------------------------------------------- #
def rho_sweep(
    rhos,
    *,
    n_pairs: int = 10,
    p_active: float = 0.10,
    t_steps: int = 60_000,
    k_slots: int = 1,
    seed: int = 0,
    reps: int = 3,
) -> list[dict]:
    """Hold (N, p) fixed and sweep latent ``rho`` in the homogeneous world; report
    the utilization predictions (naive + both heuristic feedings) vs simulated
    truth at each rho (averaged over ``reps`` seeds). Isolates how each
    prediction's error changes with rho, including the crossover where the
    latent-fed heuristic starts beating the naive model."""
    out = []
    ss = np.random.SeedSequence(seed)
    for ri, rho in enumerate(rhos):
        accs: dict[str, list[float]] = {}
        for rep in range(reps):
            rng = np.random.default_rng(ss.spawn(1)[0])
            cfg = SignalConfig(n_pairs=n_pairs, p_active=p_active, rho=float(rho),
                               structure="equicorr", t_steps=t_steps, label="sweep")
            sim = simulate(cfg, rng)
            u = utilization_predictions(sim, k_slots=k_slots)
            for key in ("p_at_least_one_sim", "p_at_least_one_naive",
                        "p_at_least_one_heuristic_latent",
                        "p_at_least_one_heuristic_binary",
                        "err_heuristic_latent", "err_heuristic_binary", "err_naive",
                        "abs_err_heuristic_latent", "abs_err_heuristic_binary",
                        "abs_err_naive", "rho_latent", "rho_binary"):
                accs.setdefault(key, []).append(u[key])
        row = {"rho": float(rho)}
        row.update({k: float(np.mean(v)) for k, v in accs.items()})
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# practical "optimal number of pairs" under slots + edge degradation
# --------------------------------------------------------------------------- #
def optimal_pairs(
    *,
    p_active: float = 0.15,
    rho: float = 0.30,
    structure: str = "equicorr",
    n_blocks: int = 1,
    loading_dispersion: float = 0.0,
    block_share: float = 0.0,
    max_pairs: int = 30,
    k_slots: int = 1,
    edge_top: float = 0.9,
    edge_decay: float = 0.06,
    t_steps: int = 40_000,
    seed: int = 0,
    reps: int = 3,
) -> dict:
    """Find the N that maximizes expected portfolio PnL/step under a simple
    edge-degradation model, comparing the *post's analytic* utilization
    (``1-(1-p)^{N_eff}`` capped by slots) against the *simulated* utilization.

    The analytic heuristic is evaluated under BOTH feedings: the measured mean
    latent correlation and the measured mean binary-activation correlation
    (each averaged over ``reps`` independent simulations per N, so
    heterogeneous loadings are properly resampled).

    For a genuinely heterogeneous run pass ``structure="factor"`` together with
    nonzero ``loading_dispersion`` (and optionally ``n_blocks``/``block_share``);
    with the defaults a "factor" structure degenerates to equicorrelation.

    Edge of the k-th best pair decays geometrically:
    ``edge_k = edge_top * (1-edge_decay)^{k-1}``. Portfolio score
    ``= avg_edge(N) * fill(N)``. We report the optimal N under each utilization
    model and the throughput cost of following each heuristic optimum.
    """
    from .breadth import binary_corr_matrix, equicorr_neff, mean_offdiag
    from .utilization import pred_post_heuristic, simulated_utilization

    edges = edge_top * (1.0 - edge_decay) ** np.arange(max_pairs)
    avg_edge = np.cumsum(edges) / np.arange(1, max_pairs + 1)

    ss = np.random.SeedSequence(seed)
    rows = []
    for n in range(1, max_pairs + 1):
        rho_lat_list, rho_bin_list = [], []
        util_sim_list, fill_one_sim_list = [], []
        for rep in range(reps):
            rng = np.random.default_rng(ss.spawn(1)[0])
            cfg = SignalConfig(n_pairs=n, p_active=p_active, rho=rho,
                               structure=structure, n_blocks=n_blocks,
                               loading_dispersion=loading_dispersion,
                               block_share=block_share,
                               t_steps=t_steps, label="optN")
            sim = simulate(cfg, rng)
            rho_lat_list.append(mean_offdiag(sim.corr_latent))
            rho_bin_list.append(
                mean_offdiag(binary_corr_matrix(sim.activations)) if n >= 2 else 0.0)
            su = simulated_utilization(sim.activations, k_slots)
            util_sim_list.append(su["utilization"])
            fill_one_sim_list.append(su["p_at_least_one"])
        rho_lat = float(np.mean(rho_lat_list))
        rho_bin = float(np.mean(rho_bin_list))

        def analytic_fill(rho_bar: float) -> float:
            # the post uses fill_efficiency = min(p_at_least_one, utilization)
            neff = equicorr_neff(n, rho_bar)
            p_one = pred_post_heuristic(n, p_active, rho_bar)
            mean_active = neff * p_active if np.isfinite(neff) else np.nan
            util = (min(mean_active, k_slots) / k_slots
                    if np.isfinite(mean_active) else np.nan)
            return min(p_one, util) if np.isfinite(util) else p_one

        fill_post_latent = analytic_fill(rho_lat)
        fill_post_binary = analytic_fill(rho_bin)
        util_sim = float(np.mean(util_sim_list))
        fill_one_sim = float(np.mean(fill_one_sim_list))
        fill_sim = min(fill_one_sim, util_sim)

        rows.append({
            "n_pairs": n,
            "avg_edge": float(avg_edge[n - 1]),
            "rho_latent": rho_lat,
            "rho_binary": rho_bin,
            "fill_post_latent": float(fill_post_latent),
            "fill_post_binary": float(fill_post_binary),
            "fill_sim": float(fill_sim),
            "score_post_latent": float(avg_edge[n - 1] * fill_post_latent),
            "score_post_binary": float(avg_edge[n - 1] * fill_post_binary),
            "score_sim": float(avg_edge[n - 1] * fill_sim),
        })

    def argmax_n(key: str) -> int:
        return int(np.nanargmax(np.array([r[key] for r in rows]))) + 1

    opt_post_latent = argmax_n("score_post_latent")
    opt_post_binary = argmax_n("score_post_binary")
    opt_sim = argmax_n("score_sim")
    score_sim_arr = np.array([r["score_sim"] for r in rows])
    best_sim = float(score_sim_arr[opt_sim - 1])

    def throughput_cost_pct(opt_post: int) -> float:
        """% of the simulated max score lost by trading at the heuristic's N."""
        if best_sim <= 0:
            return float("nan")
        return float((1.0 - score_sim_arr[opt_post - 1] / best_sim) * 100.0)

    return {
        "params": {"p_active": p_active, "rho": rho, "structure": structure,
                   "n_blocks": n_blocks, "loading_dispersion": loading_dispersion,
                   "block_share": block_share, "k_slots": k_slots,
                   "edge_top": edge_top, "edge_decay": edge_decay,
                   "t_steps": t_steps, "seed": seed, "reps": reps},
        "rows": rows,
        "optimal_n_post_latent": opt_post_latent,
        "optimal_n_post_binary": opt_post_binary,
        "optimal_n_sim": opt_sim,
        "gap_latent": opt_post_latent - opt_sim,
        "gap_binary": opt_post_binary - opt_sim,
        "throughput_cost_pct_latent": throughput_cost_pct(opt_post_latent),
        "throughput_cost_pct_binary": throughput_cost_pct(opt_post_binary),
    }


def optimal_pairs_multi(n_seeds: int = 5, *, seed: int = 0, **kwargs) -> dict:
    """Run ``optimal_pairs`` across ``n_seeds`` independent seeds and report the
    per-seed optima plus their spread. The simulated score curve is flat near
    its maximum, so a single-seed optimum is noisy; the spread quantifies that
    noise, and the throughput cost says how much following the heuristic's N
    actually loses."""
    runs = [optimal_pairs(seed=seed + 1000 * i, **kwargs) for i in range(n_seeds)]
    out = {
        "n_seeds": int(n_seeds),
        "base_seed": int(seed),
        "params": runs[0]["params"],
        "per_seed": [
            {k: r[k] for k in (
                "optimal_n_post_latent", "optimal_n_post_binary", "optimal_n_sim",
                "gap_latent", "gap_binary",
                "throughput_cost_pct_latent", "throughput_cost_pct_binary")}
            for r in runs
        ],
        # full curve of the first seed, for the figure
        "representative": runs[0],
    }
    for key in ("optimal_n_post_latent", "optimal_n_post_binary", "optimal_n_sim",
                "gap_latent", "gap_binary"):
        vals = [r[key] for r in runs]
        out[f"{key}_min"] = int(min(vals))
        out[f"{key}_max"] = int(max(vals))
        out[f"{key}_median"] = float(np.median(vals))
    for key in ("throughput_cost_pct_latent", "throughput_cost_pct_binary"):
        vals = [r[key] for r in runs]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_max"] = float(np.max(vals))
    return out
