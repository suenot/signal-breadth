"""Assert that every quantitative claim in paper/main.tex matches results/results.json.

    python scripts/check_paper_numbers.py

Each check formats a value from results.json exactly the way the paper quotes it
and asserts the resulting token appears in main.tex (and, where the paper states
an inequality or a range, asserts the underlying inequality holds). Exits
non-zero on the first group of failures, so it can gate a release.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = re.sub(r"\s+", " ", (ROOT / "paper" / "main.tex").read_text())  # normalize wraps
R = json.loads((ROOT / "results" / "results.json").read_text())

failures: list[str] = []
n_checks = 0


def check(label: str, token: str, cond: bool = True) -> None:
    """Assert ``token`` appears in main.tex (whitespace-normalized) and ``cond`` holds."""
    global n_checks
    n_checks += 1
    ok_tex = token in TEX
    if ok_tex and cond:
        print(f"  PASS  {label:58} {token!r}")
    else:
        why = [] if ok_tex else [f"token {token!r} not in main.tex"]
        if not cond:
            why.append("condition failed")
        failures.append(f"{label}: {'; '.join(why)}")
        print(f"  FAIL  {label:58} {token!r}  <-- {'; '.join(why)}")


def f1(v):  # 51.7
    return f"{v:.1f}"


def s1(v):  # +4.1 / -28.4
    return f"{v:+.1f}"


# ---------------------------------------------------------------- counts ----
print("[design counts]")
check("n experiments", "4{,}000", R["n_experiments"] == 4000)
check("n homogeneous", "1{,}995", R["n_homogeneous"] == 1995)
check("n heterogeneous", "2{,}005", R["n_heterogeneous"] == 2005)
proto = R["meta"]["protocol"]["main_batch"]
check("N grid", r"N\in\{5,8,10,15,20,30\}", proto["n_grid"] == [5, 8, 10, 15, 20, 30])
check("p grid", r"p\in\{0.05,0.10,0.15,0.25,0.45\}",
      proto["p_grid"] == [0.05, 0.10, 0.15, 0.25, 0.45])
check("beta grid", r"\beta\in\{0,0.3,0.6,0.9\}",
      proto["beta_grid"] == [0.0, 0.3, 0.6, 0.9])
check("rho sampling", r"\rho\sim\mathrm{Uniform}[0,0.7]",
      proto["rho_sampling"].startswith("uniform(0.0, 0.7)"))
check("~1000 per beta", "1{,}000",
      all(900 <= row["n"] <= 1100 for row in R["breadth_accuracy_by_beta"]))

# ------------------------------------------------- Table 1: by structure ----
print("[Table 1: breadth accuracy by structure]")
ba = R["breadth_accuracy"]
for grp, gname in [("overall", "Overall"), ("homogeneous", "Hom"), ("heterogeneous", "Het")]:
    for est, ename in [("neff_post_binary", "Neff binary"),
                       ("neff_post_latent", "Neff latent"),
                       ("enb_meucci_latent", "ENB latent")]:
        d = ba[grp][est]
        check(f"{gname} {ename} MAPE", f1(d["mape_pct"]) + r"\%")
        check(f"{gname} {ename} bias", "$" + s1(d["bias_pct"]) + r"\%$")

# ------------------------------------------------------ Table 2: by beta ----
print("[Table 2: breadth accuracy by beta]")
for row in R["breadth_accuracy_by_beta"]:
    b = row["beta"]
    check(f"beta={b} binary MAPE", f1(row["mape_neff_post_binary"]) + r"\%")
    check(f"beta={b} binary bias", "$" + s1(row["bias_neff_post_binary"]) + r"\%$")
    check(f"beta={b} latent MAPE", f1(row["mape_neff_post_latent"]) + r"\%")
    check(f"beta={b} latent bias", "$" + s1(row["bias_neff_post_latent"]) + r"\%$")
    check(f"beta={b} spread", f1(row["binary_minus_latent_spread_pp"]))
betas = sorted(r_["beta"] for r_ in R["breadth_accuracy_by_beta"])
spreads = [r_["binary_minus_latent_spread_pp"]
           for r_ in sorted(R["breadth_accuracy_by_beta"], key=lambda x: x["beta"])]
check("spread range quoted (low end)", "$" + f1(spreads[0]) + "$pp")
check("spread range quoted (high end)", "$" + f1(spreads[-1]) + "$pp")
check("abstract: binary bias at beta=0", r"$-56\%$",
      round(R["breadth_accuracy_by_beta"][0]["bias_neff_post_binary"]) == -56)
check("abstract: binary bias at beta=0.9", r"$+91\%$",
      round(spreads is not None and R["breadth_accuracy_by_beta"][3]["bias_neff_post_binary"]) == 91)
check("abstract: binary bias at beta=0.6", r"$+22\%$",
      round(R["breadth_accuracy_by_beta"][2]["bias_neff_post_binary"]) == 22)
check("abstract: latent bias at beta=0.9 (+30%)", r"$+30\%$",
      round(R["breadth_accuracy_by_beta"][3]["bias_neff_post_latent"]) == 30)

# ------------------------------------------------------------ tetrachoric ----
print("[tetrachoric / feeding gap]")
g = R["binary_latent_gap"]
check("mean latent corr", f"{g['mean_latent_corr']:.3f}")
check("mean binary corr", f"{g['mean_binary_corr']:.3f}")
check("median attenuation", f"{g['median_attenuation_ratio']:.2f}")
check("frac binary below latent", r"$98\%$",
      round(g["frac_binary_below_latent"] * 100) == 98)
check("tetrachoric MAE", f"{g['tetrachoric_pred_mae']:.3f}")
infl = g["neff_inflation_from_using_binary"]
check("inflation +56%", r"$+56\%$", round(infl["mean_pct_overcount"]) == 56)
check("mean Neff binary 5.19", f"{infl['mean_neff_post_binary']:.2f}")
check("mean Neff latent 3.43", f"{infl['mean_neff_post_latent']:.2f}")

# ------------------------------------------------------- signal-stream / rho ----
print("[vs signal streams and rho stratification]")
vs = ba["vs_realized_signal"]["neff_post_binary"]
check("vs signal MAPE 0.022%", f"{vs['mape_pct']:.3f}" + r"\%", vs["mape_pct"] < 0.05)
check("abstract 0.02% (2dp)", r"$0.02\%$", round(vs["mape_pct"], 2) == 0.02)
rb = {(row["beta"], row["rho_bin"]): row for row in R["breadth_accuracy_by_rho_beta"]}
b06_low = rb[(0.6, "(-0.001, 0.1]")]["mape_post_neff_binary"]
b06_high = rb[(0.6, "(0.45, 0.7]")]["mape_post_neff_binary"]
check("beta=0.6 low-rho 62%", r"$62\%$", round(b06_low) == 62)
check("beta=0.6 high-rho 9%", r"$9\%$", round(b06_high) == 9)
b0_low = rb[(0.0, "(-0.001, 0.1]")]["mape_post_neff_binary"]
b0_high = rb[(0.0, "(0.45, 0.7]")]["mape_post_neff_binary"]
check("beta=0 low-rho 23%", r"$23\%$", round(b0_low) == 23)
check("beta=0 high-rho 76%", r"$76\%$", round(b0_high) == 76)

# ----------------------------------------------------------------- ENB ------
print("[ENB]")
check("ENB latent ~104%", r"$104\%$", round(ba["overall"]["enb_meucci_latent"]["mape_pct"]) == 104)
check("ENB returns ~120%", r"$120\%$", round(ba["overall"]["enb_meucci_returns"]["mape_pct"]) == 120)

# ------------------------------------------------------------- capacity -----
print("[capacity, K=1 and K=3]")
u = R["utilization_error_k1"]
su = R["slot_utilization_error_k3"]
check("naive MAE 0.137", f"{u['mae_naive']:.3f}")
check("latent-fed MAE 0.209", f"{u['mae_heuristic_latent']:.3f}")
check("latent-fed bias -0.209", f"{u['bias_heuristic_latent_signed']:.3f}")
check("latent-fed beats naive 20%", r"$20\%$",
      round(u["heuristic_latent_better_than_naive_rate"] * 100) == 20)
check("binary-fed MAE 0.105", f"{u['mae_heuristic_binary']:.3f}")
check("binary-fed beats naive 68%", r"$68\%$",
      round(u["heuristic_binary_better_than_naive_rate"] * 100) == 68)
check("abstract MAE 0.21 (2dp)", "$0.21$", round(u["mae_heuristic_latent"], 2) == 0.21)
check("abstract naive 0.14 (2dp)", "$0.14$", round(u["mae_naive"], 2) == 0.14)
check("abstract binary 0.11 (2dp)", "$0.11$", round(u["mae_heuristic_binary"], 2) == 0.11)
check("Spearman latent 0.49", f"{u['spearman_abserr_vs_rho_latent']:.2f}")
check("Spearman binary 0.65", f"{u['spearman_abserr_vs_rho_binary']:.2f}")
check("p < 1e-200 both", r"$p<10^{-200}$",
      u["spearman_p_latent"] < 1e-200 and u["spearman_p_binary"] < 1e-200)
terc = {t["rho_tercile"]: t for t in u["by_rho_tercile"]}
for name in ("low_rho", "mid_rho", "high_rho"):
    t = terc[name]
    tok = (f"{t['mae_heuristic_latent']:.2f}/{t['mae_heuristic_binary']:.2f}/"
           f"{t['mae_naive']:.2f}")
    check(f"tercile {name} L/B/N", tok)
check("Np MAE 0.011", f"{su['mae_mean_active_independent']:.3f}")
check("latent mean-active MAE 2.13", f"{su['mae_mean_active_post_latent']:.2f}")
check("binary mean-active MAE 1.87", f"{su['mae_mean_active_post_binary']:.2f}")
check("latent ratio 0.31x", f"{su['mean_active_post_vs_true_ratio_latent']:.2f}" + r"\times")
check("binary ratio 0.44x", f"{su['mean_active_post_vs_true_ratio_binary']:.2f}" + r"\times")
check("sim util K=3 0.491", f"{su['mean_util_sim']:.3f}")
check("latent util K=3 0.226", f"{su['mean_util_post_latent']:.3f}")
check("binary util K=3 0.306", f"{su['mean_util_post_binary']:.3f}")

# ------------------------------------------------------------- optimal-N ----
print("[optimal-N, multi-seed]")
om, oh = R["optimal_n"], R["optimal_n_heterogeneous"]
check("hom sim range 11-13", "$11$--$13$",
      om["optimal_n_sim_min"] == 11 and om["optimal_n_sim_max"] == 13)
check("hom sim median 12", "median $12$", om["optimal_n_sim_median"] == 12)
check("hom latent-fed N=7 every seed", "$N=7$ in every",
      om["optimal_n_post_latent_min"] == 7 and om["optimal_n_post_latent_max"] == 7)
check("hom binary-fed 9-10", "$N=9$--$10$",
      om["optimal_n_post_binary_min"] == 9 and om["optimal_n_post_binary_max"] == 10)
check("hom gap latent -4..-6", "gap $-4$ to $-6$",
      om["gap_latent_min"] == -6 and om["gap_latent_max"] == -4)
check("hom gap binary -2..-4", "gap $-2$ to $-4$",
      om["gap_binary_min"] == -4 and om["gap_binary_max"] == -2)
check("het sim range 10-15", "$10$--$15$",
      oh["optimal_n_sim_min"] == 10 and oh["optimal_n_sim_max"] == 15)
check("het sim median 13", "median $13$", oh["optimal_n_sim_median"] == 13)
check("het latent-fed 7-13", "$7$--$13$",
      oh["optimal_n_post_latent_min"] == 7 and oh["optimal_n_post_latent_max"] == 13)
check("het binary-fed 8-15", "$8$--$15$",
      oh["optimal_n_post_binary_min"] == 8 and oh["optimal_n_post_binary_max"] == 15)
check("het gap latent -7..0", "gap $-7$ to $0$",
      oh["gap_latent_min"] == -7 and oh["gap_latent_max"] == 0)
check("het gap binary -5..0", "gap $-5$ to $0$",
      oh["gap_binary_min"] == -5 and oh["gap_binary_max"] == 0)
check("hom cost latent 6.9%", r"$6.9\%$",
      round(om["throughput_cost_pct_latent_mean"], 1) == 6.9)
check("het cost latent 5.1%", r"$5.1\%$",
      round(oh["throughput_cost_pct_latent_mean"], 1) == 5.1)
check("hom cost binary 1.7%", r"$1.7\%$",
      round(om["throughput_cost_pct_binary_mean"], 1) == 1.7)
check("het cost binary 1.1%", r"$1.1\%$",
      round(oh["throughput_cost_pct_binary_mean"], 1) == 1.1)
rep = om["representative"]
check("fig caption rep latent N=7", r"$N\!=\!7$", rep["optimal_n_post_latent"] == 7)
check("fig caption rep binary N=9", r"$N\!=\!9$", rep["optimal_n_post_binary"] == 9)
check("fig caption rep sim N=11", r"$N\!=\!11$", rep["optimal_n_sim"] == 11)
check("abstract <=7% cost", r"$\le 7\%$",
      max(om["throughput_cost_pct_latent_mean"],
          oh["throughput_cost_pct_latent_mean"]) <= 7.0)

# --------------------------------------------------------------- rho sweep --
print("[rho sweep crossover]")
sw = R["rho_sweep"]
abs_lat = [abs(row["err_heuristic_latent"]) for row in sw]
abs_nai = [abs(row["err_naive"]) for row in sw]
abs_bin = [abs(row["err_heuristic_binary"]) for row in sw]
rhos = [row["rho"] for row in sw]
cross = next(i for i in range(1, len(sw)) if abs_lat[i] < abs_nai[i] and abs_lat[i - 1] >= abs_nai[i - 1])
check("crossover at rho~0.5", r"\rho\approx0.5", abs(rhos[cross] - 0.5) < 0.051)
peak = rhos[abs_lat.index(max(abs_lat))]
check("latent-fed error peaks near rho~0.3", r"\rho\approx0.3", abs(peak - 0.3) < 0.11)
check("binary-fed below naive for rho>=0.1", r"$\rho\ge0.1$",
      all(abs_bin[i] < abs_nai[i] for i in range(1, len(sw))))

# -------------------------------------------------------------------- done --
print(f"\n{n_checks} checks, {len(failures)} failures.")
if failures:
    for f_ in failures:
        print("FAIL:", f_)
    sys.exit(1)
print("ALL PAPER NUMBERS MATCH results.json")
