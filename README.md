# How Many Correlated Signals Actually Diversify?

A short, reproducible research paper (and the code behind it) that tests a popular
desk shortcut: the equicorrelation **effective number** `N_eff = N/(1+(N-1)ρ̄)`,
used both to discount the diversification of `N` correlated trading signals and to
size a slot-limited execution orchestrator via `1-(1-p)^N_eff`.

> **Headline findings** (4,000 simulated portfolios with known ground truth;
> the return-factor loading β is an explicit swept design parameter
> ∈ {0, 0.3, 0.6, 0.9}, not a hidden constant):
> 1. As a **diversification discount** for the binary signal streams themselves,
>    `N_eff` is essentially **exact** (0.02% error). For the **PnL** streams its
>    accuracy — and even the *sign* of its bias — depends on β: binary-fed bias
>    swings from −56% (β=0) to +91% (β=0.9), +22% at the central β=0.6. The
>    β-robust effect is the **feeding gap**: tetrachoric attenuation (latent 0.32
>    → binary 0.18, predicted to 0.018 MAE) makes any binary-fed estimate claim
>    ~56% more breadth than the latent-fed one.
> 2. Meucci's **Effective Number of Bets** measures a *different* quantity (factor
>    breadth) and is **not** interchangeable with equal-weight variance reduction.
> 3. As a **capacity model**: the mean number of simultaneously active signals is
>    `N·p` by linearity *regardless* of correlation, so the `N_eff·p` load
>    estimate is structurally wrong (0.31×/0.44× the true load, latent/binary-fed).
>    The `1-(1-p)^N_eff` fill heuristic undershoots: latent-fed it is worse than
>    assuming independence (MAE 0.21 vs 0.14, beats naive in 20% of cases);
>    binary-fed it is better (MAE 0.11, 68%) — yet both bias the implied "optimal
>    number of pairs" low (by 4–6 / 2–4 at the homogeneous design point), at a
>    modest ≤7% throughput cost because the score curve is flat.

Bottom line: use `N_eff` as a conservative breadth discount fed the *latent*
(or return) correlation; size slot budgets from `N·p` and the distribution of
the active count, never from `N_eff`. This is a de-commercialized,
experimentally-validated audit of a [marketmaker.cc](https://marketmaker.cc)
blog post by the same author.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # ~1 min -> results/results.json + records.csv
python -m signal_experiments.figures # -> paper/figures/*.pdf
```

Deterministic given the seeds in `scripts/run_all.py`. `--quick` runs a small batch.

## Build the paper

```bash
cd paper && tectonic main.tex        # -> main.pdf  (install: brew install tectonic)
```

## Layout

```
signal_experiments/
  model.py        # latent-Gaussian generator of correlated binary signals + PnL
  breadth.py      # equicorr N_eff, Meucci ENB, realized variance-reduction, tetrachoric
  utilization.py  # slot orchestrator: naive vs N_eff-heuristic vs correlated-Bernoulli truth
  simulate.py     # one experiment + Monte-Carlo batches, rho sweep, optimal-N
  analysis.py     # breadth accuracy, binary-latent gap, utilization error
  figures.py      # the paper's figures
scripts/run_all.py
tests/            # pytest sanity checks (17 tests)
paper/            # main.tex, refs.bib, figures/, compiled main.pdf
results/          # results.json + records.csv (generated)
docs/             # related-work notes
```

## Tests

```bash
python -m pytest -q     # 17 sanity tests
```

## License

Code: [MIT](LICENSE). Paper text and figures: CC BY 4.0.
