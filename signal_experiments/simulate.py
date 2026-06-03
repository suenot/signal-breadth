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
    the three utilization predictions vs simulated truth at each rho (averaged
    over ``reps`` seeds). Isolates how the heuristic error grows with rho."""
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
                        "p_at_least_one_heuristic", "err_heuristic", "err_naive",
                        "abs_err_heuristic", "abs_err_naive", "rho_latent"):
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

    Edge of the k-th best pair decays geometrically: ``edge_k = edge_top * (1-edge_decay)^{k-1}``.
    Portfolio score ``= avg_edge(N) * utilization(N)``. We report the optimal N
    under each utilization model and whether they agree.
    """
    edges = edge_top * (1.0 - edge_decay) ** np.arange(max_pairs)
    avg_edge = np.cumsum(edges) / np.arange(1, max_pairs + 1)

    ss = np.random.SeedSequence(seed)
    rows = []
    for idx in range(1, max_pairs + 1):
        n = idx
        # analytic (post) utilization with N_eff and slots
        from .breadth import equicorr_neff
        from .utilization import pred_post_heuristic

        neff = equicorr_neff(n, rho)
        p_one_post = pred_post_heuristic(n, p_active, rho)
        mean_active_post = neff * p_active if np.isfinite(neff) else np.nan
        util_post = min(mean_active_post, k_slots) / k_slots if np.isfinite(mean_active_post) else np.nan
        # the post uses fill_efficiency = min(p_at_least_one, utilization)
        fill_post = min(p_one_post, util_post) if np.isfinite(util_post) else p_one_post

        # simulated utilization (averaged over reps)
        util_sim_list, fill_one_sim_list = [], []
        for rep in range(reps):
            rng = np.random.default_rng(ss.spawn(1)[0])
            cfg = SignalConfig(n_pairs=n, p_active=p_active, rho=rho,
                               structure=structure, t_steps=t_steps, label="optN")
            sim = simulate(cfg, rng)
            from .utilization import simulated_utilization

            su = simulated_utilization(sim.activations, k_slots)
            util_sim_list.append(su["utilization"])
            fill_one_sim_list.append(su["p_at_least_one"])
        util_sim = float(np.mean(util_sim_list))
        fill_one_sim = float(np.mean(fill_one_sim_list))
        fill_sim = min(fill_one_sim, util_sim)

        rows.append({
            "n_pairs": n,
            "avg_edge": float(avg_edge[idx - 1]),
            "fill_post": float(fill_post),
            "fill_sim": float(fill_sim),
            "score_post": float(avg_edge[idx - 1] * fill_post),
            "score_sim": float(avg_edge[idx - 1] * fill_sim),
        })

    scores_post = np.array([r["score_post"] for r in rows])
    scores_sim = np.array([r["score_sim"] for r in rows])
    opt_post = int(np.nanargmax(scores_post)) + 1
    opt_sim = int(np.nanargmax(scores_sim)) + 1
    return {
        "params": {"p_active": p_active, "rho": rho, "structure": structure,
                   "k_slots": k_slots, "edge_top": edge_top, "edge_decay": edge_decay},
        "rows": rows,
        "optimal_n_post": opt_post,
        "optimal_n_sim": opt_sim,
        "agree": int(opt_post == opt_sim),
        "n_gap": opt_post - opt_sim,
    }
