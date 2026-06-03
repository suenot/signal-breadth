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
from signal_experiments.simulate import optimal_pairs, rho_sweep, run_batch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n = 400 if args.quick else 4000
    t_steps = 12_000 if args.quick else 30_000
    sweep_reps = 2 if args.quick else 5
    sweep_t = 30_000 if args.quick else 120_000
    opt_t = 20_000 if args.quick else 60_000
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

    print("[3/4] optimal-N (slots + edge degradation) ...", flush=True)
    opt_main = optimal_pairs(p_active=0.15, rho=0.30, structure="equicorr",
                             max_pairs=30, k_slots=1, t_steps=opt_t, seed=404,
                             reps=sweep_reps)
    # a heterogeneous comparison point
    opt_het = optimal_pairs(p_active=0.15, rho=0.30, structure="factor",
                            max_pairs=30, k_slots=1, t_steps=opt_t, seed=505,
                            reps=sweep_reps)

    print("[4/4] summaries ...", flush=True)
    summary = A.summarize(df)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_experiments": n,
            "t_steps": t_steps,
            "seeds": {"batch": 101, "sweep": 303, "optimal_n": 404, "optimal_n_het": 505},
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

    print("\nBreadth accuracy (MAPE% / signed bias% vs realized PnL effective N):")
    for est in ["neff_post_binary", "neff_post_latent", "enb_meucci_latent", "enb_meucci_returns"]:
        o, h, e = ba["overall"][est], ba["homogeneous"][est], ba["heterogeneous"][est]
        print(f"  {est:20} overall {o['mape_pct']:5.0f}/{o['bias_pct']:+5.0f}  "
              f"hom {h['mape_pct']:5.0f}/{h['bias_pct']:+5.0f}  "
              f"het {e['mape_pct']:5.0f}/{e['bias_pct']:+5.0f}")

    g = summary["binary_latent_gap"]
    print(f"\nBinary vs latent correlation: latent {g['mean_latent_corr']:.3f} -> "
          f"binary {g['mean_binary_corr']:.3f} (gap {g['mean_gap_latent_minus_binary']:.3f}, "
          f"attenuation {g['median_attenuation_ratio']:.2f}); tetrachoric pred MAE "
          f"{g['tetrachoric_pred_mae']:.4f}")
    print(f"  using binary corr inflates N_eff by "
          f"{g['neff_inflation_from_using_binary']['mean_pct_overcount']:.0f}% vs latent corr")

    u = summary["utilization_error_k1"]
    print(f"\nUtilization P(>=1 active) error (single slot): heuristic MAE {u['mae_heuristic']:.3f}, "
          f"naive MAE {u['mae_naive']:.3f}, heuristic bias {u['bias_heuristic_signed']:+.3f}")
    print(f"  |heuristic error| Spearman vs rho = {u['spearman_abserr_vs_rho']:.2f} "
          f"(p={u['spearman_p']:.1e}); by rho tercile:")
    for t in u["by_rho_tercile"]:
        print(f"    {t['rho_tercile']:9} rho~{t['mean_rho']:.2f}  "
              f"MAE heuristic {t['mae_heuristic']:.3f}  naive {t['mae_naive']:.3f}")

    su = summary["slot_utilization_error_k3"]
    print(f"\nSlot utilization (k=3): post mean {su['mean_util_post']:.3f} vs simulated "
          f"{su['mean_util_sim']:.3f} (MAE {su['mae_util_post']:.3f}, bias {su['bias_util_post']:+.3f})")
    print(f"  E[# active]: post uses N_eff*p = {su['mean_active_post_vs_true_ratio']:.2f}x the true N*p "
          f"(post MAE {su['mae_mean_active_post']:.3f} vs N*p MAE {su['mae_mean_active_independent']:.3f})")

    print(f"\nOptimal-N (p=0.15, rho=0.30, 1 slot, decaying edge):")
    print(f"  homogeneous:   post says N={opt_main['optimal_n_post']:2d}, "
          f"simulation says N={opt_main['optimal_n_sim']:2d} "
          f"(gap {opt_main['n_gap']:+d})")
    print(f"  heterogeneous: post says N={opt_het['optimal_n_post']:2d}, "
          f"simulation says N={opt_het['optimal_n_sim']:2d} "
          f"(gap {opt_het['n_gap']:+d})")


if __name__ == "__main__":
    main()
