"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + CSV
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below. No wall-clock / randomness leaks into
results. Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_experiments import __version__
from signal_experiments import analysis as A
from signal_experiments.model import (
    SAMPLED_BETA_GRID,
    SAMPLED_N_GRID,
    SAMPLED_P_GRID,
    SAMPLED_RHO_RANGE,
)
from signal_experiments.simulate import optimal_pairs_multi, rho_sweep, run_batch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# genuinely heterogeneous config for the optimal-N comparison point (nonzero
# loading dispersion + blocks; defaults would silently degenerate to equicorr)
HET_OPT_PARAMS = {"n_blocks": 3, "loading_dispersion": 0.35, "block_share": 0.10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n = 400 if args.quick else 4000
    t_steps = 12_000 if args.quick else 30_000
    sweep_reps = 2 if args.quick else 5
    sweep_t = 30_000 if args.quick else 120_000
    opt_t = 20_000 if args.quick else 60_000
    opt_reps = 2 if args.quick else 3
    opt_seeds = 2 if args.quick else 5
    RESULTS.mkdir(exist_ok=True)

    print(f"[1/4] main batch (n={n}, T={t_steps}) ...", flush=True)
    recs = run_batch(n, structure="random", seed=101, t_steps=t_steps,
                     k_slots_list=(1, 3), progress_every=max(1, n // 4))
    df = A.to_frame(recs)
    df.to_csv(RESULTS / "records.csv", index=False)

    print("[2/4] rho sweep (utilization heuristic) ...", flush=True)
    rhos = np.linspace(0.0, 0.8, 9)
    sweep = rho_sweep(rhos, n_pairs=10, p_active=0.10, t_steps=sweep_t,
                      k_slots=1, seed=303, reps=sweep_reps)

    print("[3/4] optimal-N (slots + edge degradation, multi-seed) ...", flush=True)
    opt_main = optimal_pairs_multi(
        opt_seeds, seed=404, p_active=0.15, rho=0.30, structure="equicorr",
        max_pairs=30, k_slots=1, t_steps=opt_t, reps=opt_reps)
    # a genuinely heterogeneous comparison point (dispersion + blocks)
    opt_het = optimal_pairs_multi(
        opt_seeds, seed=505, p_active=0.15, rho=0.30, structure="factor",
        max_pairs=30, k_slots=1, t_steps=opt_t, reps=opt_reps, **HET_OPT_PARAMS)

    print("[4/4] summaries ...", flush=True)
    summary = A.summarize(df)

    def counts(col: str) -> dict:
        return {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()}

    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_experiments": n,
            "t_steps": t_steps,
            "seeds": {"batch": 101, "sweep": 303, "optimal_n": 404, "optimal_n_het": 505},
            # the ACTUAL sampled design, reported verbatim in the paper (no
            # rounded-off "ranges" that were never sampled)
            "protocol": {
                "main_batch": {
                    "n_experiments": n,
                    "t_steps": t_steps,
                    "seed": 101,
                    "n_grid": list(SAMPLED_N_GRID),
                    "p_grid": list(SAMPLED_P_GRID),
                    "beta_grid": list(SAMPLED_BETA_GRID),
                    "rho_sampling": (
                        f"uniform({SAMPLED_RHO_RANGE[0]}, {SAMPLED_RHO_RANGE[1]}), "
                        "clipped to the PSD-valid equicorrelation range"),
                    "structure_mix": "equicorr vs heterogeneous one-factor(+blocks), 50/50",
                    "observed_counts": {
                        "n_pairs": counts("cfg_n_pairs"),
                        "p_active": counts("cfg_p_active"),
                        "beta": counts("cfg_factor_return_beta"),
                        "structure": counts("cfg_structure"),
                    },
                },
                "rho_sweep": {"rhos": [float(r) for r in rhos], "n_pairs": 10,
                              "p_active": 0.10, "t_steps": sweep_t,
                              "reps": sweep_reps, "k_slots": 1, "seed": 303},
                "optimal_n": {"n_seeds": opt_seeds, "reps_per_seed": opt_reps,
                              "t_steps": opt_t, "max_pairs": 30, "k_slots": 1,
                              "p_active": 0.15, "rho": 0.30,
                              "het_params": HET_OPT_PARAMS},
            },
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        **summary,
        "rho_sweep": sweep,
        "optimal_n": opt_main,
        "optimal_n_heterogeneous": opt_het,
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS/'results.json'} and records.csv.")

    # ---- headline numbers to stdout -----------------------------------
    ba = summary["breadth_accuracy"]
    print("\n--- HEADLINE NUMBERS ---")
    print(f"experiments: {summary['n_experiments']} "
          f"({summary['n_homogeneous']} homogeneous, {summary['n_heterogeneous']} heterogeneous)")

    print("\nBreadth accuracy (MAPE% / signed bias% vs realized PnL effective N, "
          "marginal over the beta grid):")
    for est in ["neff_post_binary", "neff_post_latent", "enb_meucci_latent", "enb_meucci_returns"]:
        o, h, e = ba["overall"][est], ba["homogeneous"][est], ba["heterogeneous"][est]
        print(f"  {est:20} overall {o['mape_pct']:5.0f}/{o['bias_pct']:+5.0f}  "
              f"hom {h['mape_pct']:5.0f}/{h['bias_pct']:+5.0f}  "
              f"het {e['mape_pct']:5.0f}/{e['bias_pct']:+5.0f}")
    vs = ba["vs_realized_signal"]
    print(f"  vs SIGNAL-stream effective N: binary-fed N_eff MAPE "
          f"{vs['neff_post_binary']['mape_pct']:.3f}% (essentially exact)")

    print("\nBreadth accuracy by return-factor loading beta "
          "(bias% binary-fed / latent-fed, spread in pp):")
    for row in summary["breadth_accuracy_by_beta"]:
        print(f"  beta={row['beta']:.1f} (n={row['n']:4d})  "
              f"binary {row['bias_neff_post_binary']:+6.1f}% (MAPE {row['mape_neff_post_binary']:5.1f}%)  "
              f"latent {row['bias_neff_post_latent']:+6.1f}% (MAPE {row['mape_neff_post_latent']:5.1f}%)  "
              f"spread {row['binary_minus_latent_spread_pp']:5.1f}pp")

    print("\nBreadth accuracy by latent-rho bin (overall, MAPE% binary-fed):")
    for row in summary["breadth_accuracy_by_rho"]:
        if row["structure"] == "overall":
            print(f"  rho {row['rho_bin']:14} (n={row['n']:4d})  "
                  f"binary {row['mape_post_neff_binary']:5.1f}%  "
                  f"latent {row['mape_post_neff_latent']:5.1f}%")

    g = summary["binary_latent_gap"]
    print(f"\nBinary vs latent correlation: latent {g['mean_latent_corr']:.3f} -> "
          f"binary {g['mean_binary_corr']:.3f} (gap {g['mean_gap_latent_minus_binary']:.3f}, "
          f"median attenuation {g['median_attenuation_ratio']:.2f}); tetrachoric pred MAE "
          f"{g['tetrachoric_pred_mae']:.4f}")
    print(f"  using binary corr inflates N_eff by "
          f"{g['neff_inflation_from_using_binary']['mean_pct_overcount']:.0f}% vs latent corr")

    u = summary["utilization_error_k1"]
    print(f"\nUtilization P(>=1 active) error (single slot): naive MAE {u['mae_naive']:.3f}")
    for v in ("latent", "binary"):
        print(f"  heuristic ({v}-fed): MAE {u[f'mae_heuristic_{v}']:.3f}, "
              f"bias {u[f'bias_heuristic_{v}_signed']:+.3f}, "
              f"beats naive {u[f'heuristic_{v}_better_than_naive_rate']*100:.0f}% of cases, "
              f"Spearman(|err|, rho) {u[f'spearman_abserr_vs_rho_{v}']:.2f} "
              f"(p={u[f'spearman_p_{v}']:.1e})")
    print("  by rho tercile (MAE latent-fed / binary-fed / naive):")
    for t in u["by_rho_tercile"]:
        print(f"    {t['rho_tercile']:9} rho~{t['mean_rho']:.2f}  "
              f"{t['mae_heuristic_latent']:.3f} / {t['mae_heuristic_binary']:.3f} / "
              f"{t['mae_naive']:.3f}")

    su = summary["slot_utilization_error_k3"]
    print(f"\nSlot utilization (k=3): simulated {su['mean_util_sim']:.3f}; "
          f"post latent-fed {su['mean_util_post_latent']:.3f} "
          f"(MAE {su['mae_util_post_latent']:.3f}), "
          f"binary-fed {su['mean_util_post_binary']:.3f} "
          f"(MAE {su['mae_util_post_binary']:.3f})")
    print(f"  E[# active] ratio to true N*p: latent-fed "
          f"{su['mean_active_post_vs_true_ratio_latent']:.2f}x, binary-fed "
          f"{su['mean_active_post_vs_true_ratio_binary']:.2f}x "
          f"(N*p itself: MAE {su['mae_mean_active_independent']:.3f})")

    print(f"\nOptimal-N (p=0.15, rho=0.30, 1 slot, decaying edge; "
          f"{opt_main['n_seeds']} seeds x {opt_main['params']['reps']} reps):")
    for name, o in [("homogeneous", opt_main), ("heterogeneous", opt_het)]:
        seeds_sim = [r["optimal_n_sim"] for r in o["per_seed"]]
        seeds_lat = [r["optimal_n_post_latent"] for r in o["per_seed"]]
        seeds_bin = [r["optimal_n_post_binary"] for r in o["per_seed"]]
        print(f"  {name:13} sim N* {seeds_sim} (median {o['optimal_n_sim_median']:.0f}), "
              f"post latent-fed {seeds_lat}, binary-fed {seeds_bin}")
        print(f"  {'':13} gap latent {o['gap_latent_min']}..{o['gap_latent_max']}, "
              f"binary {o['gap_binary_min']}..{o['gap_binary_max']}; "
              f"throughput cost latent {o['throughput_cost_pct_latent_mean']:.1f}% "
              f"binary {o['throughput_cost_pct_binary_mean']:.1f}%")


if __name__ == "__main__":
    main()
