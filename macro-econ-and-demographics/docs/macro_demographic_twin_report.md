# The Demographic-Debt Relay
## A state-space analysis of fertility, modernisation and debt in four countries (1950–2020)

---

## Summary
This report summarises the analysis in `training/notebooks/analysis.ipynb`, which studies how fertility relates to urbanisation, schooling, child mortality, public social spending and debt in **Sweden, Japan, the United States and Italy**.

The methods are **Augmented Dickey-Fuller (ADF) tests, Granger causality tests, Engle-Granger cointegration tests**, and a per-country **Dynamic Factor state-space model** (two latent factors, AR(1) factors, AR(1) errors, estimated with the Kalman filter on standardised annual data).

What the analysis supports:
* **Predictive ordering between two cross-country averages.** Using average total fertility across the Human Fertility Database populations and average public + private debt/GDP across the JST countries (1891–2020, first-differenced), past fertility changes improve the prediction of debt changes at lags 1–4 (F-test $p$ = 0.0016, 0.0006, 0.0015, 0.0043). Past debt changes do not improve the prediction of fertility ($p$ = 0.9309, 0.7255, 0.8519, 0.6485). This is predictive precedence in this sample, not a mechanism. In levels, the two averages are negatively correlated (Pearson −0.70, Spearman −0.75).
* **No detected long-run equilibrium** between public social spending and fertility in any of the four countries.
* **Some country-specific predictive links** between the modernisation series and fertility. Several would not survive a correction for the number of tests run.
* **Four distinct fitted latent trajectories**, read in this report as structural archetypes. That reading is an interpretation of the fitted model.

These results are consistent with the hypothesis that fertility decline and debt accumulation are linked, with fertility moving first. The analysis does not test that hypothesis causally, and it does not show that rising debt is a demographic crisis in disguise. The debt series used throughout is a single global indicator, not each country's own debt (see Limitations).

---

## 1. Social spending and fertility: cointegration tests
Historically, children functioned as an informal pension policy and labor force. The introduction of the modern welfare state socialized the benefits of reproduction while keeping costs private. *(Not traceable to any notebook output.)*

Engle-Granger cointegration tests between public social spending (% of GDP) and fertility were run on the longest window available for each country:

| Country | Window | Years | $p$ |
| --- | --- | --- | --- |
| Sweden | 1891–2020 | 130 | 0.9169 |
| United States | 1933–2020 | 88 | 0.8976 |
| Japan | 1947–2020 | 74 | 0.9504 |
| Italy | 1954–2020 | 67 | 0.9745 |

None of the tests rejects the null hypothesis of no cointegration. This is an absence of a detected long-run equilibrium, not evidence that social spending and fertility are unrelated. Engle-Granger tests have limited power on samples of this length, especially on interpolated series, so a long-run relationship could exist and go undetected.

The result also does not point to a specific alternative, such as both series being driven by a shared modernisation process. That remains a hypothesis. Short-run Granger tests on the differenced series found nothing either: none of the 24 tests (four countries, both directions, lags 1–3) was significant at 5 %. The lowest was $p$ = 0.2116, for social spending → fertility in the United States at lag 1.

---

## 2. Modernisation series and fertility: Granger tests (1950–2020)
For each country, the notebook first-differences fertility, urbanisation, mean years of schooling and child mortality, then runs Granger tests in both directions at lags 1–3. Differencing is meant to remove trends. The per-country differenced series were not themselves checked with ADF tests; the notebook's ADF checks cover only the cross-country fertility and debt averages.

A significant result below means that past values of one series improve the prediction of the other beyond that series' own past, in this sample. It does not establish that one drives the other.

* **Child mortality → fertility.** Significant in Japan (lag 1 $p$ < 0.0001, lag 2 $p$ = 0.0023), Sweden (lag 3 $p$ = 0.0002) and Italy (lag 3 $p$ = 0.0342), but not in the United States (lowest $p$ = 0.0507, lag 2). The ordering is consistent with the hypothesis that better child survival reduces "insurance births", but the test does not identify that mechanism. The reverse ordering is also significant in Sweden (fertility → child mortality, lag 1 $p$ = 0.0158) and the United States (lag 1 $p$ < 0.0001), so the precedence does not run one way everywhere.
* **Urbanisation → fertility (Japan).** Lag 1 $p$ = 0.0001, lag 2 $p$ = 0.0077. One hypothesis consistent with this, though not demonstrated by it, is that postwar migration into dense cities raised the cost of large families.
* **Schooling → fertility (United States).** Lag 1 $p$ = 0.0441; lags 2 and 3 are not significant ($p$ = 0.1389, 0.1854). This is a marginal single-lag result that does not survive a multiple-testing correction. The idea that higher attainment shortens the early reproductive window is a hypothesis this result neither confirms nor rules out.
* **Fertility → urbanisation.** Significant at all three lags in the United States ($p$ = 0.0256, 0.0209, 0.0355) and Sweden ($p$ = 0.0134, 0.0247, 0.0335). The idea that smaller families let younger cohorts stay in high-wage urban centres is likewise a hypothesis, not a result.

**Number of tests.** This set of tests contains 72 (4 countries × 3 series × 2 directions × 3 lags). Fifteen are significant at 5 %, where roughly 3–4 would be expected by chance if no relationships existed; tests at neighbouring lags are correlated, so that expectation is approximate. Under a Bonferroni threshold of 0.05 / 72 ≈ 0.0007, four remain:
- child mortality → fertility in Japan (lag 1) and Sweden (lag 3)
- urbanisation → fertility in Japan (lag 1)
- fertility → child mortality in the United States (lag 1)

A companion set tests debt against the same three series (72 tests). Four were significant at 5 %, about what chance alone would produce, and one passes the Bonferroni threshold: debt → child mortality in Japan at lag 3 ($p$ = 0.0006). Because the debt series is the same global indicator for every country, none of these results is country-specific.

---

## 3. Latent factors: four fitted trajectories
For each country, a Dynamic Factor model summarises five standardised series (fertility, debt, urbanisation, schooling, child mortality) with two latent factors. The fitting window is 1950–2020, or 1954–2020 for Italy.

**What the factors are.** The model returns two unlabelled factors, and their sign and rotation are not identified: flipping a factor's sign along with its loadings, or rotating the pair, fits the data equally well. Calling Factor 1 "Modernisation" and Factor 2 "Systemic Strain" is an interpretation of the loadings, not a finding of the model, and the labels are used below only as shorthand.

The factors are also not independent of each other. The fitted transition matrix couples them; for Sweden, the off-diagonal coefficients are 0.149 and −0.388. The notebook prints loadings only for Italy. There, both factors load on all five series with the same signs and small, similar weights (debt, for example, loads 0.026 on Factor 1 and 0.024 on Factor 2), so the two labels are not cleanly separated. In every country, the debt series the factors load on is the global indicator, not that country's own debt.

**What the charts show.** The notebook plots the factor time series only for Italy. For all four countries it draws phase portraits (Factor 1 against Factor 2), which have no year axis. Year-specific statements below for Sweden, Japan and the United States cannot be read off a notebook chart, and they are marked as such.

The notebook also fits a VAR(1) to each country's smoothed factors. Every one is stable, with spectral radius below one: Sweden 0.959, Japan 0.993, United States 0.981, Italy 0.976. The implied half-lives are 16.64, 104.48, 35.94 and 28.84 years. These describe a second model fitted to estimated factors and carry no uncertainty bounds.

### A. Sweden, read as a managed system with cyclical strain
* **Factor 1 ("Modernisation"):** Smooth ascent to a comfortable, stable modern plateau by 2000. *(Not traceable to any notebook chart.)*
* **Factor 2 ("Systemic Strain"):** Moves in clean, distinct cyclical waves. It peaked sharply during the 1990 banking crisis before structural welfare overhauls successfully discharged the systemic tension back to baseline. *(Not traceable to any notebook chart.)*

### B. Japan, read as a completed regime transition
* **Factor 1 ("Modernisation"):** A steep, permanent structural drop from 1950 to 1980, capturing the rapid burning-out of its agrarian past. *(Not traceable to any notebook chart.)*
* **Factor 2 ("Systemic Strain"):** A deep U-shaped valley that bottoms out in 1985 before entering an unyielding, post-1990 upward climb. *(Not traceable to any notebook chart.)* This factor loads on the global debt indicator, so its rise says nothing about Japanese state debt issuance specifically.

### C. United States, read as synchronised movement
* **Factors 1 and 2:** In the phase portrait, the two factors move together along a nearly straight path.
* **After 2000:** The moment the dot-com bubble burst (2000), both factors hit a hard structural ceiling, permanently freezing America's modernization index and fixing its systemic financial strain at a historic peak. *(Not traceable to any notebook chart.)*

### D. Italy, read as a divergence between the two factors
* **Factor 1 ("Modernisation"):** Rises to a peak around 1978–79 and declines steadily through 2020.
* **Factor 2 ("Systemic Strain"):** Rises steadily from a low in the mid-1960s to its highest value in 2020. The rise begins before Factor 1 peaks, and it is closer to linear than exponential. Italy's loadings are small, so large movements in the factors correspond to modest movements in the standardised series.

These archetypes describe the shapes of the fitted trajectories. They are not established structural types, and none of them is evidence about a country's own debt.

**Extrapolating the fitted dynamics to 2050.** The notebook multiplies each country's 2020 latent state by its fitted transition matrix thirty times. Where the result lands is a property of that matrix's eigenvalues. They are all below one in modulus but close to it, so repeated multiplication pulls the state slowly back toward zero. The 2050 point is an extrapolation of the fitted dynamics, not a forecast: it has no uncertainty bounds, it comes from fits that did not converge, and no out-of-sample validation supports it.

---

## 4. Impulse responses from the fitted Sweden model
The notebook computes 20-year impulse responses from Sweden's fitted model. An impulse response shows how the estimated system propagates a one-off shock under its own fitted dynamics. It is not a policy experiment and carries no causal identification, so it says nothing about what an intervention on fertility would achieve.

Two details of the computation matter for reading it:
* **The shock is to the first latent factor, not to the fertility series.** The call `impulse_responses(impulse=0)` shocks the first state innovation, which is Factor 1, although the notebook describes the shock as a positive fertility shock. Sweden's fertility series loads negatively on Factor 1, so fertility falls on impact.
* **The panel labelled as the two hidden layers shows observed series, not factors.** It plots the first two columns of the response table: the responses of fertility and debt.

In standardised units, the debt response is positive on impact, turns negative after about six years, reaches about −0.35 SD around year 11, and returns to roughly zero by year 18. That is a damped oscillation of the fitted system. The debt series is the global indicator, so this describes how the Sweden model propagates a factor shock into that indicator, not how Swedish debt would respond to anything. The notebook computes no impulse responses for Japan, the United States or Italy.

---

## 5. Limitations
* **Debt is one global series, not each country's debt.** Every debt input from the JST data onward is the cross-country mean of public plus private debt/GDP, a single series shared by all four countries. It is also not sovereign debt alone, because private loans are included. Every statement about a particular country's debt, including the Sweden impulse response, is really about this shared indicator. This is the most important limitation of the analysis.
* **Averages with changing membership.** The fertility-debt Granger result averages fertility across all Human Fertility Database populations present in each year, and debt across the JST countries. Which populations are present changes over time; in 1891 the fertility average is Sweden alone. Part of the movement in these averages therefore reflects which countries are in the data.
* **Multiple testing.** The notebook runs 176 Granger tests (8 on the cross-country averages, 24 on social spending, 72 on the modernisation series, 72 on debt) with no correction. Some significant results are expected by chance; Section 2 shows which survive a Bonferroni threshold.
* **Interpolation.** Series are reindexed to an annual timeline and filled: fertility, social spending and debt linearly; urbanisation, schooling and child mortality quadratically. Interpolated points are not observations. Smoothing makes consecutive changes more alike, which inflates test statistics in Granger and ADF tests and overstates the effective sample size of the factor model.
* **Sample size.** Each country has 71 annual observations (Italy 67). The factor model estimates, on five series, ten loadings, a two-by-two factor transition matrix, and an AR(1) coefficient and variance for each series' error.
* **Convergence.** None of the four country fits reports convergence within the 250-iteration limit. `twin-train` reproduces the notebook's fits exactly and records `converged: false` for each country in the artifact manifest. The factors, impulse responses and extrapolations all come from these unconverged fits.
* **No out-of-sample evaluation in the analysis.** Nothing in the notebook tests the model on data it was not fitted to. The training pipeline has since added a backtest (fit through 2000, evaluate 2001–2020); in a Sweden run, the model's forecasts were worse than holding the 2000 values flat, for all five series.
* **Reporting of small p-values.** The notebook prints p-values below 0.00005 as 0.0000; they are written here as < 0.0001.

---

## Conclusions
The analysis finds:
- fertility changes preceding debt changes, in the predictive sense, in cross-country averages
- country-specific predictive links between child mortality, urbanisation and fertility, of which a few remain after correcting for the number of tests
- no detected cointegration between social spending and fertility
- four distinct fitted latent trajectories that can be read as structural archetypes

The chain "modernisation changes incentives → families shrink → pay-as-you-go safety nets come under strain → debt expands" is a hypothesis that parts of these results are consistent with. The analysis cannot test it: it uses no pension data and no country-level debt, has no identification strategy, and has no out-of-sample evaluation. No policy was modelled, so nothing here indicates which fiscal responses would fail or succeed in any country.

The notebook's section 6 ("Den cybernetiska syntesen") and its epilogue give a cybernetic reading of these results: interconnected nodes that compensate for one another's failures, a spinning coin approaching a tipping point, and four cybernetic remedies. That material is an essay that the figures illustrate, not a finding of the model.
