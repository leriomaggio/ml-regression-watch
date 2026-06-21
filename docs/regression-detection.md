# Regression detection

How the harness decides whether a configuration has become slower. This is one
stage of the [methodology](methodology.md); it consumes the latency samples
produced by [benchmarking](benchmarking.md) and shares its baseline with
[numerical validation](numerical-validation.md). How it is gated in CI is covered
in [continuous integration](ci-and-artifacts.md).

Regression detection treats the per-repeat latency measurements of a configuration as a
sample rather than a single number, and compares the current sample against the stored
baseline sample with a one-sided Mann-Whitney U test (`scipy.stats.mannwhitneyu`,
`alternative="greater"`).

A configuration is flagged as a regression only when both conditions hold:

1. The test is significant (p < 0.05): the current latencies are stochastically greater
   than the baseline latencies.
2. The relative increase in median latency exceeds a configurable margin (default 5
   percent).

Requiring both avoids flagging differences that are statistically detectable but
practically irrelevant. The report includes the rank-biserial effect size, derived from
the same U statistic and bounded in [-1, 1], alongside the relative change.
