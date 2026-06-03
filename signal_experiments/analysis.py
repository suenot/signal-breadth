"""Turn a batch of experiment records into the paper's quantitative results.

Four questions, all answered with measured numbers (no fabrication):

  1. **Breadth accuracy.** How well does the post's equicorrelation ``N_eff``
     match the ground-truth realized effective N, vs Meucci's ENB? Split by
     homogeneous (equicorrelation) vs heterogeneous (factor/block) structure --
     N_eff should be near-exact in the former and biased in the latter.
  2. **Binary vs latent correlation.** Quantify the attenuation: binary-signal
     correlation is systematically below latent correlation, so the breadth you
     get from signal correlation differs from latent/return correlation.
  3. **Utilization heuristic error.** Error of ``1-(1-p)^{N_eff}`` vs the
     simulated correlated-Bernoulli truth, and how it grows with rho.
  4. **Optimal-N.** Does the optimal number of pairs predicted by the post's
     analytic utilization match the one from simulation?

All returns are JSON-able (dicts / lists / records).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# 1. breadth-estimator accuracy
# --------------------------------------------------------------------------- #
def _mape(series: pd.Series) -> float:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.mean(np.abs(s)) * 100.0) if len(s) else float("nan")


def _bias(series: pd.Series) -> float:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.mean(s) * 100.0) if len(s) else float("nan")


def breadth_accuracy(df: pd.DataFrame) -> dict:
    """Mean-abs-% error and signed bias of each breadth estimator vs the realized
    PnL effective N, overall and split by homogeneous/heterogeneous structure.

    ``neff_post_binary`` is exactly what the blog post computes (equicorrelation
    formula fed the binary signal correlation); ``neff_post_latent`` feeds it the
    latent correlation; ``enb_meucci_returns`` is the principled baseline.
    """
    estimators = {
        "neff_post_binary": "relerr_neff_post_binary",
        "neff_post_latent": "relerr_neff_post_latent",
        "enb_meucci_latent": "relerr_enb_latent",
        "enb_meucci_returns": "relerr_enb_returns",
    }
    out = {"overall": {}, "homogeneous": {}, "heterogeneous": {}}
    groups = {
        "overall": df,
        "homogeneous": df[df["is_homogeneous"] == 1],
        "heterogeneous": df[df["is_homogeneous"] == 0],
    }
    for gname, g in groups.items():
        for name, col in estimators.items():
            out[gname][name] = {
                "mape_pct": _mape(g[col]),
                "bias_pct": _bias(g[col]),
                "n": int(g[col].replace([np.inf, -np.inf], np.nan).notna().sum()),
            }
    # also vs the binary-signal-stream effective N (a second target)
    out["vs_realized_signal"] = {
        "neff_post_binary": {"mape_pct": _mape(df["relerr_neff_post_binary_vs_signal"]),
                             "bias_pct": _bias(df["relerr_neff_post_binary_vs_signal"])},
        "enb_meucci_latent": {"mape_pct": _mape(df["relerr_enb_latent_vs_signal"]),
                              "bias_pct": _bias(df["relerr_enb_latent_vs_signal"])},
    }
    return out


def breadth_accuracy_by_rho(df: pd.DataFrame, bins=(0.0, 0.1, 0.25, 0.45, 0.7, 1.0)) -> list[dict]:
    """MAPE of post N_eff vs ENB across latent-rho bins, split by structure, vs
    the realized PnL effective N."""
    rows = []
    for struct, mask in [("homogeneous", df["is_homogeneous"] == 1),
                         ("heterogeneous", df["is_homogeneous"] == 0)]:
        h = df[mask].copy()
        h["rho_bin"] = pd.cut(h["rho_latent"], bins=list(bins), include_lowest=True)
        for b, g in h.groupby("rho_bin", observed=True):
            if len(g) == 0:
                continue
            rows.append({
                "structure": struct,
                "rho_bin": str(b),
                "n": int(len(g)),
                "mape_post_neff_binary": _mape(g["relerr_neff_post_binary"]),
                "mape_enb_latent": _mape(g["relerr_enb_latent"]),
            })
    return rows


# --------------------------------------------------------------------------- #
# 2. binary vs latent correlation gap
# --------------------------------------------------------------------------- #
def binary_latent_gap(df: pd.DataFrame) -> dict:
    """How far binary-signal correlation sits below latent correlation, and the
    accuracy of the tetrachoric prediction of the binary correlation."""
    gap = df["corr_gap_latent_minus_binary"]
    atten = df["rho_binary"] / df["rho_latent"].replace(0, np.nan)
    # tetrachoric prediction error: predicted phi vs realized binary corr
    pred_err = df["phi_pred_tetrachoric"] - df["rho_binary"]
    pos = df[df["rho_latent"] > 0.02]
    return {
        "mean_latent_corr": float(df["rho_latent"].mean()),
        "mean_binary_corr": float(df["rho_binary"].mean()),
        "mean_gap_latent_minus_binary": float(gap.mean()),
        "median_attenuation_ratio": float(atten.replace([np.inf, -np.inf], np.nan).dropna().median()),
        "frac_binary_below_latent": float((df["rho_binary"] < df["rho_latent"] - 1e-6).mean()),
        "tetrachoric_pred_mae": float(np.mean(np.abs(pred_err.dropna()))),
        "neff_inflation_from_using_binary": {
            # using binary corr makes rho smaller -> N_eff larger -> over-counts breadth
            "mean_neff_post_latent": float(pos["neff_post_latent"].mean()),
            "mean_neff_post_binary": float(pos["neff_post_binary"].mean()),
            "mean_pct_overcount": float(
                ((pos["neff_post_binary"] - pos["neff_post_latent"])
                 / pos["neff_post_latent"]).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100.0),
        },
    }


# --------------------------------------------------------------------------- #
# 3. utilization heuristic error
# --------------------------------------------------------------------------- #
def utilization_error(df: pd.DataFrame, k: int = 1) -> dict:
    """Error of naive and post-heuristic P(>=1 active) vs simulated truth, overall
    and by latent-rho tercile, for ``k`` slots."""
    pfx = f"util_"
    sfx = f"_k{k}"
    abs_h = df[f"{pfx}abs_err_heuristic{sfx}"]
    abs_n = df[f"{pfx}abs_err_naive{sfx}"]
    sgn_h = df[f"{pfx}err_heuristic{sfx}"]
    rho = df[f"{pfx}rho_latent{sfx}"]

    terc = pd.qcut(rho, 3, labels=["low_rho", "mid_rho", "high_rho"], duplicates="drop")
    by_terc = []
    for t, g in df.groupby(terc, observed=True):
        by_terc.append({
            "rho_tercile": str(t),
            "mean_rho": float(g[f"{pfx}rho_latent{sfx}"].mean()),
            "mae_heuristic": float(g[f"{pfx}abs_err_heuristic{sfx}"].mean()),
            "mae_naive": float(g[f"{pfx}abs_err_naive{sfx}"].mean()),
            "bias_heuristic": float(g[f"{pfx}err_heuristic{sfx}"].mean()),
        })
    # correlation of |heuristic error| with rho (does it grow with rho?)
    ok = np.isfinite(abs_h) & np.isfinite(rho)
    rho_corr, p = stats.spearmanr(rho[ok], abs_h[ok])
    return {
        "k_slots": k,
        "mae_heuristic": float(abs_h.mean()),
        "mae_naive": float(abs_n.mean()),
        "bias_heuristic_signed": float(sgn_h.mean()),
        "heuristic_better_than_naive_rate": float((abs_h < abs_n).mean()),
        "spearman_abserr_vs_rho": float(rho_corr),
        "spearman_p": float(p),
        "by_rho_tercile": by_terc,
    }


def slot_utilization_error(df: pd.DataFrame, k: int = 3) -> dict:
    """Error of the post's analytic slot utilization ``min(N_eff*p, K)/K`` against
    the simulated ``E[min(active, K)]/K`` for ``k`` slots.

    The post's ``E[active] = N_eff*p`` is wrong by construction: by linearity of
    expectation the true mean number active is ``N*p`` regardless of correlation,
    so the post systematically under-states demand. With slot capping the effect
    on utilization is more subtle; we measure it directly."""
    pfx, sfx = "util_", f"_k{k}"
    util_sim = df[f"{pfx}utilization_sim{sfx}"]
    mean_active_sim = df[f"{pfx}mean_active_sim{sfx}"]
    mean_active_post = df[f"{pfx}mean_active_post{sfx}"]  # N_eff * p
    mean_active_indep = df[f"{pfx}mean_active_independent{sfx}"]  # N * p (truth)
    # post utilization estimate: min(N_eff*p, K)/K
    util_post = np.minimum(mean_active_post, k) / k
    err = util_post - util_sim
    return {
        "k_slots": k,
        "mean_util_sim": float(util_sim.mean()),
        "mean_util_post": float(util_post.mean()),
        "mae_util_post": float(np.abs(err).mean()),
        "bias_util_post": float(err.mean()),
        # the E[active] story (linearity): post N_eff*p vs true N*p vs simulated
        "mean_active_post_vs_true_ratio": float((mean_active_post / mean_active_indep).mean()),
        "mae_mean_active_post": float(np.abs(mean_active_post - mean_active_sim).mean()),
        "mae_mean_active_independent": float(np.abs(mean_active_indep - mean_active_sim).mean()),
    }


# --------------------------------------------------------------------------- #
# top-level summary
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame) -> dict:
    return {
        "n_experiments": int(len(df)),
        "n_homogeneous": int((df["is_homogeneous"] == 1).sum()),
        "n_heterogeneous": int((df["is_homogeneous"] == 0).sum()),
        "breadth_accuracy": breadth_accuracy(df),
        "breadth_accuracy_by_rho": breadth_accuracy_by_rho(df),
        "binary_latent_gap": binary_latent_gap(df),
        "utilization_error_k1": utilization_error(df, k=1),
        "slot_utilization_error_k3": slot_utilization_error(df, k=3),
    }
