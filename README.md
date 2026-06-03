# How Many Correlated Signals Actually Diversify?

A short, reproducible research paper (and the code behind it) that tests a popular
desk shortcut: the equicorrelation **effective number** `N_eff = N/(1+(N-1)ρ̄)`,
used both to discount the diversification of `N` correlated trading signals and to
size a slot-limited execution orchestrator via `1-(1-p)^N_eff`.

> **Headline findings** (4,000 simulated portfolios with known ground truth):
> 1. As a **diversification discount**, `N_eff` is rough (~25% error) and
>    **sign-sensitive**: fed the *binary* signal correlation it over-counts by
>    +20%; fed the *latent* correlation it under-counts by −18%. The gap is the
>    tetrachoric attenuation of dichotomized signals (latent 0.33 → binary 0.18,
>    predicted to 0.018 MAE).
> 2. Meucci's **Effective Number of Bets** measures a *different* quantity (factor
>    breadth) and is **not** interchangeable with equal-weight variance reduction.
> 3. As a **capacity model** it is **structurally wrong**: the mean number of
>    simultaneously active signals is `N·p` by linearity *regardless* of
>    correlation, so `1-(1-p)^N_eff` undershoots — error **growing with ρ̄** — and
>    the implied "optimal number of pairs" is biased low by 5–6.

Bottom line: use `N_eff` to discount breadth (fed the *latent* correlation);
never to size a slot budget. This is a de-commercialized, experimentally-validated
rewrite of a [marketmaker.cc](https://marketmaker.cc) blog post.

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
tests/            # pytest sanity checks (12 tests)
paper/            # main.tex, refs.bib, figures/, compiled main.pdf
results/          # results.json + records.csv (generated)
docs/             # related-work notes
```

## Tests

```bash
python -m pytest -q     # 12 sanity tests
```

## License

Code: [MIT](LICENSE). Paper text and figures: CC BY 4.0.
