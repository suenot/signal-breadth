"""Generate the paper's figures (vector PDF) from the saved results.

    python -m signal_experiments.figures      # writes paper/figures/*.pdf

Reads results/results.json, results/records.csv, and recomputes the small
illustrative panels (setup, rho-sweep, optimal-N) deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .breadth import binary_corr_matrix, equicorr_neff, mean_offdiag, phi_from_tetrachoric
from .model import SignalConfig, canonical_configs, simulate
from .simulate import optimal_pairs, rho_sweep
from .utilization import pred_naive_independent, pred_post_heuristic

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TRUE, C_POST, C_NAIVE, C_ENB, C_PTS = "#1f3b73", "#c0392b", "#9aa6b2", "#2e8b57", "#e0a458"


# --------------------------------------------------------------------------- #
# Fig 1: setup -- latent factor -> correlated binary activations
# --------------------------------------------------------------------------- #
def fig_setup(path: Path) -> None:
    cfg = SignalConfig(n_pairs=6, p_active=0.12, rho=0.45, structure="equicorr", t_steps=4000)
    sim = simulate(cfg, np.random.default_rng(3))
    window = slice(0, 160)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))

    # (a) latent Gaussians for a few pairs + the shared threshold
    ax = axes[0]
    z = sim.latent[window]
    for i in range(4):
        ax.plot(z[:, i], lw=0.8, alpha=0.8, color=plt.cm.viridis(i / 4))
    ax.axhline(sim.threshold, color="k", lw=1.2, ls="--", label=r"activation threshold $\Phi^{-1}(1-p)$")
    ax.set_title("(a) correlated latent signals")
    ax.set_xlabel("time step"); ax.set_ylabel("latent value $z_{i,t}$")
    ax.legend(fontsize=6.5, loc="lower right")

    # (b) resulting binary activations (raster) -- note the clustering in time
    ax = axes[1]
    act = sim.activations[window].T
    ax.imshow(act, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_title("(b) binary activations cluster in time")
    ax.set_xlabel("time step"); ax.set_ylabel("pair $i$")
    ax.set_yticks(range(cfg.n_pairs))

    # (c) latent vs binary correlation -> attenuation
    ax = axes[2]
    cb = binary_corr_matrix(sim.activations)
    rho_lat = cfg.rho
    rho_bin = mean_offdiag(cb)
    rhos = np.linspace(0.0, 0.95, 50)
    phis = [phi_from_tetrachoric(r, cfg.p_active) for r in rhos]
    ax.plot(rhos, rhos, color=C_PTS, lw=1.2, ls=":", label="identity")
    ax.plot(rhos, phis, color=C_POST, lw=1.8, label=r"binary $\phi$ (tetrachoric)")
    ax.scatter([rho_lat], [rho_bin], color=C_TRUE, zorder=5, s=30,
               label=f"simulated (p={cfg.p_active})")
    ax.set_title("(c) binary signal corr $<$ latent corr")
    ax.set_xlabel(r"latent correlation $\rho$")
    ax.set_ylabel(r"binary-signal correlation")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: N_eff accuracy vs realized effective-N, hom vs het, with ENB
# --------------------------------------------------------------------------- #
def fig_breadth_accuracy(path: Path, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    # (a) scatter: predicted vs realized effective N (the post's binary N_eff)
    ax = axes[0]
    hom = df[df["is_homogeneous"] == 1]
    het = df[df["is_homogeneous"] == 0]
    truth_col = "realized_neff_pnl"
    ax.scatter(hom[truth_col], hom["neff_post_binary"], s=10, alpha=0.5,
               color=C_TRUE, label="homogeneous")
    ax.scatter(het[truth_col], het["neff_post_binary"], s=10, alpha=0.5,
               color=C_POST, label="heterogeneous")
    # full data range -- no axis clipping of any point
    lim = [0, 1.02 * max(df[truth_col].max(), df["neff_post_binary"].max())]
    ax.plot(lim, lim, "k--", lw=1, label="exact")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(r"(a) post's $N_{\mathrm{eff}}$ (binary corr) vs truth")
    ax.set_xlabel("realized effective $N$ (PnL variance reduction)")
    ax.set_ylabel(r"$N_{\mathrm{eff}} = N/(1+(N-1)\bar\rho_{\mathrm{bin}})$")
    ax.legend(fontsize=7, loc="upper left")

    # (b) MAPE bars per estimator, hom vs het
    ax = axes[1]
    ests = ["neff_post_binary", "neff_post_latent", "enb_meucci_latent"]
    labels = [r"$N_{\mathrm{eff}}$ (binary)", r"$N_{\mathrm{eff}}$ (latent)", "Meucci ENB"]
    err_cols = {"neff_post_binary": "relerr_neff_post_binary",
                "neff_post_latent": "relerr_neff_post_latent",
                "enb_meucci_latent": "relerr_enb_latent"}

    def mape(sub, col):
        s = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
        return float(np.mean(np.abs(s)) * 100)

    x = np.arange(len(ests)); w = 0.38
    hom_v = [mape(hom, err_cols[e]) for e in ests]
    het_v = [mape(het, err_cols[e]) for e in ests]
    ax.bar(x - w / 2, hom_v, w, color=C_TRUE, label="homogeneous")
    ax.bar(x + w / 2, het_v, w, color=C_POST, label="heterogeneous")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("mean abs. % error vs realized $N$")
    ax.set_title("(b) accuracy by estimator and structure")
    ax.legend(fontsize=7)
    for xi, v in zip(x - w / 2, hom_v):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=6.5)
    for xi, v in zip(x + w / 2, het_v):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=6.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: utilization -- simulated vs naive vs heuristic across rho
# --------------------------------------------------------------------------- #
def fig_utilization(path: Path, sweep: list[dict]) -> None:
    """Both heuristic feedings vs naive vs simulated truth, and the |error|
    curves including the crossover where the latent-fed heuristic starts
    beating the naive model at high rho."""
    sw = pd.DataFrame(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    ax = axes[0]
    ax.plot(sw["rho"], sw["p_at_least_one_naive"], "o-", color=C_NAIVE,
            label=r"naive $1-(1-p)^N$")
    ax.plot(sw["rho"], sw["p_at_least_one_heuristic_latent"], "s-", color=C_POST,
            label=r"post, latent-fed $N_{\mathrm{eff}}$")
    ax.plot(sw["rho"], sw["p_at_least_one_heuristic_binary"], "d-", color=C_PTS,
            label=r"post, binary-fed $N_{\mathrm{eff}}$")
    ax.plot(sw["rho"], sw["p_at_least_one_sim"], "^-", color=C_TRUE,
            label="simulated (correlated)")
    ax.set_title(r"(a) $P(\geq 1$ active$)$ vs latent $\rho$")
    ax.set_xlabel(r"latent correlation $\rho$")
    ax.set_ylabel(r"$P(\geq 1$ active$)$")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    abs_lat = np.abs(sw["err_heuristic_latent"])
    abs_naive = np.abs(sw["err_naive"])
    ax.plot(sw["rho"], abs_lat, "s-", color=C_POST, label="post, latent-fed")
    ax.plot(sw["rho"], np.abs(sw["err_heuristic_binary"]), "d-", color=C_PTS,
            label="post, binary-fed")
    ax.plot(sw["rho"], abs_naive, "o-", color=C_NAIVE, label="naive")
    # mark the crossover where the latent-fed heuristic starts beating naive
    d = np.asarray(abs_lat) - np.asarray(abs_naive)
    worse = np.where(d > 0)[0]  # rhos where naive is better
    if len(worse) and worse[-1] + 1 < len(d) and d[worse[-1] + 1] < 0:
        rho_c = float(sw["rho"].iloc[worse[-1] + 1])
        ax.axvline(rho_c, color="k", lw=0.8, ls=":")
        ax.text(rho_c + 0.01, float(np.max(abs_naive)) * 0.55,
                f"crossover\n$\\rho\\approx{rho_c:.1f}$", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(r"(b) $|$error$|$ vs $\rho$: latent-fed crosses below naive")
    ax.set_xlabel(r"latent correlation $\rho$")
    ax.set_ylabel(r"$|$predicted $-$ simulated$|$")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: practical optimal-N -- post analytic vs simulated
# --------------------------------------------------------------------------- #
def fig_optimal_n(path: Path, opt: dict) -> None:
    """Representative seed of the multi-seed optimal-N exercise, showing both
    heuristic feedings against the simulated truth."""
    rep = opt.get("representative", opt)  # accept multi-seed or single result
    rows = pd.DataFrame(rep["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    ax = axes[0]
    ax.plot(rows["n_pairs"], rows["fill_post_latent"], "s--", color=C_POST,
            label="post fill (latent-fed)")
    ax.plot(rows["n_pairs"], rows["fill_post_binary"], "d--", color=C_PTS,
            label="post fill (binary-fed)")
    ax.plot(rows["n_pairs"], rows["fill_sim"], "^-", color=C_TRUE,
            label="simulated fill efficiency")
    ax.plot(rows["n_pairs"], rows["avg_edge"], "o-", color=C_ENB, label="avg edge (decaying)")
    ax.set_title("(a) fill efficiency vs avg edge")
    ax.set_xlabel("number of pairs $N$"); ax.set_ylabel("value")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(rows["n_pairs"], rows["score_post_latent"], "s--", color=C_POST,
            label=f"post latent-fed (opt $N={rep['optimal_n_post_latent']}$)")
    ax.plot(rows["n_pairs"], rows["score_post_binary"], "d--", color=C_PTS,
            label=f"post binary-fed (opt $N={rep['optimal_n_post_binary']}$)")
    ax.plot(rows["n_pairs"], rows["score_sim"], "^-", color=C_TRUE,
            label=f"simulated (opt $N={rep['optimal_n_sim']}$)")
    ax.axvline(rep["optimal_n_post_latent"], color=C_POST, lw=0.8, ls=":")
    ax.axvline(rep["optimal_n_post_binary"], color=C_PTS, lw=0.8, ls=":")
    ax.axvline(rep["optimal_n_sim"], color=C_TRUE, lw=0.8, ls=":")
    ax.set_title("(b) portfolio score = avg edge $\\times$ fill")
    ax.set_xlabel("number of pairs $N$")
    ax.set_ylabel("expected portfolio PnL / step")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "records.csv")
    results = json.loads((RESULTS / "results.json").read_text())

    # recompute the small illustrative panels deterministically if missing
    sweep = results.get("rho_sweep") or rho_sweep(
        np.linspace(0.0, 0.8, 9), n_pairs=10, p_active=0.10, seed=303, reps=3)
    opt = results.get("optimal_n") or optimal_pairs(
        p_active=0.15, rho=0.30, max_pairs=30, seed=404)  # fig handles either shape

    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_breadth_accuracy(FIGDIR / "fig_breadth_accuracy.pdf", df)
    fig_utilization(FIGDIR / "fig_utilization.pdf", sweep)
    fig_optimal_n(FIGDIR / "fig_optimal_n.pdf", opt)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
