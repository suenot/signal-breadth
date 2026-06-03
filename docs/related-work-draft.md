# Related Work (draft)

The question of how many genuinely distinct bets a portfolio contains is as old as
modern portfolio theory itself. \citet{markowitz1952portfolio} formalized
diversification as the systematic exploitation of imperfect correlation: the
variance of a portfolio depends not on the number of holdings but on the
covariance structure among them, so that adding positively correlated assets
yields diminishing risk-reduction. The naive count of $N$ positions therefore
overstates true diversification whenever holdings co-move. Our paper studies the
analogous overstatement for a collection of $N$ *trading signals* rather than
$N$ assets, and asks whether a simple closed-form correction recovers the
effective count.

A substantial literature has sought scalar summaries of "how diversified" a
portfolio is. \citet{meucci2009managing} proposed the Effective Number of Bets
(ENB), obtained by rotating the portfolio onto its uncorrelated principal
portfolios, normalizing their risk contributions into a *diversification
distribution*, and taking the exponential of that distribution's Shannon entropy.
The ENB equals $N$ when risk is spread evenly across $N$ uncorrelated sources and
collapses toward one as risk concentrates, giving a principled, correlation-aware
"effective number." The ENB is the natural ground-truth benchmark for any
coarser approximation: it is exact (up to the choice of torsion/rotation) but
requires the full covariance matrix and an eigendecomposition. A recurring theme
in this work, anticipated by the random-matrix critique of empirical
correlation matrices \citep{laloux1999noise} and the shrinkage remedy of
\citet{ledoit2004honey}, is that the covariance matrix on which such measures
depend is itself estimated with substantial noise, motivating interest in
robust, low-parameter approximations.

The active-management literature attacks the same problem from the angle of
*breadth*. \citet{grinold1989fundamental} introduced the Fundamental Law of
Active Management, $\mathrm{IR} \approx \mathrm{IC}\cdot\sqrt{BR}$, relating the
information ratio to the information coefficient and the breadth $BR$, defined as
the number of *independent* bets taken per period; the textbook treatment in
\citet{grinold2000active} develops the law and its assumptions in detail. The
independence requirement is precisely the load-bearing assumption for our
problem: when signals or bets are correlated, the raw count of decisions
overstates breadth, and the realized information ratio falls short of the naive
prediction. \citet{clarke2002portfolio} generalized the law to constrained
portfolios, introducing the *transfer coefficient* that quantifies how
imperfectly a manager's signals are translated into positions, and
\citet{clarke2006fundamental} extended the framework to a full (non-diagonal)
covariance matrix, making explicit that breadth becomes difficult to define and
shrinks once security returns---or, equivalently, the signals driving them---are
correlated. These works establish, qualitatively and in specific settings, that
correlation erodes effective breadth; they do not, however, supply or validate a
single closed-form effective-count formula for an arbitrary number of binary
signals.

A candidate for such a formula exists in classical statistics. Under
equicorrelation---all pairwise correlations equal to a common $\bar\rho$---the
variance of the sample mean of $N$ unit-variance variables is inflated by the
factor $1+(N-1)\bar\rho$, so that the *effective sample size* is
$N_{\mathrm{eff}} = N / \bigl(1+(N-1)\bar\rho\bigr)$. This is the design-effect /
effective-sample-size result of survey sampling \citep{kish1965survey}, where
$1+(N-1)\bar\rho$ is the variance-inflation factor induced by intraclass
correlation within clusters. The formula is attractive precisely because it is
parameter-frugal: it needs only $N$ and a single average correlation, sidestepping
the noisy full-matrix estimation that troubles ENB-style measures. Whether this
single-parameter equicorrelation approximation accurately predicts the ENB and
the realized diversification of a portfolio of correlated *trading signals* is,
to our knowledge, untested.

Two features of trading signals complicate the direct transplant of the
equicorrelation result. First, actionable signals are typically *binary* (in or
out, long or flat) rather than Gaussian, and the relevant correlation is the
correlation between Bernoulli indicators, not their latent drivers. The
distinction between the latent correlation of underlying continuous variables and
the observed correlation of their dichotomized realizations is the classical
tetrachoric problem \citep{pearson1900tetrachoric}: thresholding a bivariate
normal attenuates the observed correlation in a nonlinear, base-rate-dependent
way, so the $\bar\rho$ that enters $N_{\mathrm{eff}}$ is not interchangeable with
the correlation of the signals' continuous precursors. Second, dependence among
financial variables is not fully captured by linear correlation: the stylized
facts of asset returns \citep{cont2001empirical} and the copula-based critiques
of correlation as a dependence measure \citep{embrechts2002correlation} both warn
that tail co-movement and nonlinear dependence can diverge sharply from what a
single $\bar\rho$ conveys. A controlled validation of $N_{\mathrm{eff}}$ must
therefore be explicit about which correlation it uses and under what dependence
structure it is being asked to hold.

The gap this paper addresses is now precise. Prior work supplies (i) an exact but
data-hungry diversification benchmark, the ENB \citep{meucci2009managing}; (ii) a
breadth concept that is known to degrade under correlation but is not reduced to a
validated closed form \citep{grinold1989fundamental,clarke2006fundamental}; and
(iii) a parameter-frugal equicorrelation effective-count formula from statistics
\citep{kish1965survey} whose applicability to binary signals and to slot-limited
execution has not been established. We close this gap with controlled simulation:
treating the ENB and a correlated-Bernoulli capacity model as ground truth, we
measure how well $N_{\mathrm{eff}} = N/(1+(N-1)\bar\rho)$ predicts the realized
effective breadth and the orchestrator (slot-limited execution) capacity of a
portfolio of $N$ correlated trading signals, taking explicit account of the
binary/tetrachoric distinction between latent and observed correlation.
