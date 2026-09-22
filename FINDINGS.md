# What LLM leaderboards hide

A day spent pulling LLM benchmark data apart programmatically — 653 models, ~89 fields each.
These are the findings that survived checking. The theme: **every individual number can be
correct while the conclusion you draw from it is wrong.**

Figures are Artificial Analysis's public data as of 2026-09-22. The tool behind them is
[in this repo](README.md).

---

## 1. Nearly half the leaderboard is noise

The catalogue publishes a 95% confidence interval for one evaluation (aa-briefcase). Across
the 72 models with enough data to compare, **31 of the 66 adjacent pairs among those that
publish an interval overlap** — about half.

```
claude-fable-5-1 (1678)  and  claude-opus-5 (1673)     overlap
claude-opus-5    (1673)  and  grok-4-7 (1657)          overlap
qwen3-8-max      (1640)  and  muse-spark-1-3 (1597)    separated -- a real gap
```

Merging the intervals rather than ranking them, **43 of the 72 share a tied quality rank**
(41 of those also carry a composite score; the other two are missing a different axis —
finding #9 in miniature).

**So:** before paying attention to a rank, check the interval. A 65-point gap with disjoint
intervals is a real difference. A 2-point gap with overlapping intervals is a coin flip the
leaderboard will happily present as #1 vs #2.

## 2. The hallucination rate inverts when read alone

Read as "share of answers that are made up", it gives you the ranking exactly backwards. It
is the share of **wrong** answers that were confabulations rather than abstentions — so a
model that says "I don't know" constantly scores beautifully.

| Model | accuracy | hallucination rate | confabulations per 100 questions |
|---|---|---|---|
| GLM-5.3 | 0.339 | 29.6% | **19.6** |
| GLM 5.3 Flash | 0.275 | 27.6% | **20.0** |
| Gemini 3.8 Flash | 0.546 | 55.2% | 25.1 |
| DeepSeek V4.1 Flash | 0.464 | 96.5% | 51.7 |
| GPT-5.6 Luna | 0.427 | 92.6% | 53.1 |
| `command-a-plus` | 0.089 | **14.2%** | 12.9 |

`command-a-plus` has the *lowest* rate in the set because it answers wrong by abstaining, at
8.9% accuracy. It isn't honest, it's quiet. And by raw rate GLM 5.3 Flash looks worst of the
well-known models at 27.6%; by actual confabulations it is roughly **2.5× better** than Luna
or DeepSeek.

**So:** the absolute figure is `(1 − accuracy) × hallucination_rate`. Never read the rate
without the accuracy beside it.

## 3. Price rankings reverse with your workload

"Price per million tokens" is one number covering very different bills, because input, output
and cached input cost wildly different amounts and the mix depends on what you're doing.

| Mix | Cheapest of the five |
|---|---|
| `0:3:1` (balanced) | GLM 5.3 Flash, $0.238 |
| `100:1:1` (cache-heavy agent loop) | **DeepSeek V4.1 Flash, $0.021** |

Rather than guessing your mix, **solve for the crossover**: DeepSeek's cache price
($0.006/M) undercuts GLM 5.3 Flash's ($0.026/M) 4.3×, and at 3:1 input:output the two tie at
a **93.5% cache-hit rate**. So "DeepSeek is cheapest at 100:1:1" is true and practically
irrelevant — GLM 5.3 Flash wins at every realistic mix.

**So:** compute the price at your own mix. If you don't know your cache-hit rate, you don't
know which model is cheaper.

## 4. Two-thirds of the catalogue is retired or guessed at

- **381 of 656 models are deprecated.** Most carry a pointer to their replacement.
- **496 of 656 intelligence indices are estimated**, not measured. The estimate is labelled
  in the data; a table of scores doesn't show you which rows are which.

**So:** a large fraction of any "compare all models" exercise compares things that are
retired or imputed. Filter both before you start.

## 5. The composite hides opposite strengths

`--evals` ranks within each of the 10 evals behind the index — not across raw scores, which
are on incomparable scales (`aa-briefcase` ~1460, `automationbench` ~0.60):

| eval | GLM-5.3 | GLM 5.3 Flash | Gemini 3.8 | DS V4.1 Flash | GPT-5.6 Luna |
|---|---|---|---|---|---|
| aa-briefcase | #1 | #2 | #5 | #3 | #4 |
| long-context-reasoning | **#5** | #4 | #3 | **#1** | #2 |
| automationbench | #2 | #3 | #4 | #1 | #5 |
| critpt | #2 | #4 | #3 | #5 | **#1** |
| gdp-pdf | **#5** | #3 | #2 | #4 | **#1** |
| gdpval | #1 | #2 | #5 | #3 | #4 |
| humanitys-last-exam | #2 | #3 | **#1** | #5 | #4 |
| omniscience | #2 | #3 | **#1** | #4 | #5 |
| scicode | #1 | #5 | #2 | #4 | #3 |
| terminalbench-4-0 | #1 | #2 | #4 | #3 | #5 |
| **wins** | **4** | **0** | **2** | **2** | **2** |

GLM-5.3 tops the composite (44.78) and wins 4 of 10, but is **dead last on long-context
reasoning and gdp-pdf**. And weighting the evals moves the winner outright:

| Emphasis | Winner |
|---|---|
| equal weights | GLM-5.3 |
| coding (`terminalbench-4-0`:3, `scicode`:2, `critpt`:1) | GLM-5.3 |
| knowledge (`humanitys-last-exam`:3, `omniscience`:3) | **Gemini 3.8 Flash** |
| long-context (`aa-long-context-reasoning`:3, `gdp-pdf`:1) | **GPT-5.6 Luna** |
| enterprise (`gdp-pdf`:3, `gdpval-aa`:2, `aa-briefcase`:2) | **GLM 5.3 Flash** |

GPT-5.6 Luna is **last** on the composite (37.32) and **first** under long-context weighting.
GLM 5.3 Flash wins **zero** individual evals yet wins the enterprise weighting, because it is
#2 or #3 on all three. Normalising rewards consistency over spiky excellence, which is the
opposite of what a wins-count tells you.

**So:** decompose the composite, and re-run it under your own emphasis. If that is your
workload, the composite ranking is actively wrong.

## 6. Effort level is a hidden axis

`effort.label` is `low|medium|high|xhigh|max`, and one model ships as several slugs —
`gemini-3-8-flash` exists as `-low` (33.45), `-medium` (39.77) and `high` (40.93). Comparing
`max` against `high` is not apples-to-apples, and ranking the raw catalogue counts one model
four times. `trustworthy()` collapses these.

**So:** collapse variants per family before ranking, or you're comparing settings, not models.

## 7. A median speed without its spread is not a speed

`outputSpeedVariance` carries `p05/q25/median/q75/p95`, and `hostModelCount` says how many
providers back it. Gemini 3.8 Flash's 343.5 tok/s spans 230–456 across **4 hosts**; GLM-5.3's
61.0 spans 30–94 across **22 hosts**.

**So:** the faster number is the less reliable one. Quote the spread and the sample size, or
don't quote it.

## 8. Some latency figures measure nothing

Latency is split into input / reasoning / answer. Some reasoning models report **zero
reasoning time** — the thinking is folded into the input bucket:

| Model | total | input | reasoning | answer |
|---|---|---|---|---|
| GLM-5.3 | 44.08 | 3.12 | 32.76 | 8.19 |
| DeepSeek V4.1 Flash | 12.00 | 0.92 | 8.86 | 2.22 |
| Gemini 3.8 Flash | 16.47 | **15.01** | **0.00** | 1.46 |
| GPT-5.6 Luna | 126.43 | **123.28** | **0.00** | 3.15 |

That 123-second "time to first chunk" isn't latency, it's untimed reasoning in the wrong
column. A breakdown can sum correctly while one term is zero — that is mis-attribution, not
absence.

**So:** if a reasoning model reports 0s of reasoning time, its latency is not comparable to a
model that reports it. Don't chart it.

## 9. Missing data gets scored as good data — a bug I wrote myself

**6 of the 72 comparable models have no published price at all** — open-weights entries with
zero hosts. They pass every quality filter and still cannot be cost-ranked.

That is also what caused the bug. The ranking combines axes (quality, cost, speed, stability)
by averaging, and those six have no cost axis — so they were averaged over the axes they *did*
have. A model measured on one axis was compared against models measured on three: a
quality-only model ranked #1 would have scored 1.00 and appeared to beat a model that was
top-3 on everything.

The fix was to refuse to score such models and say why. The general rule is worth more than
the bug: **when data is missing, the honest output is a gap, not a number.** Averaging over
what you have rewards the absence.

It survived every test I had written, because all of them used models with complete data.
Edge cases with *missing fields* are a different test class from malformed input, and I had
only written the second kind.

## 10. The winner depends on the scenario you didn't sweep

Every ranking embeds assumptions — what you weight, what your traffic mix is. Sweep them and
the answer moves.

Across the five reference models, 5 price mixes × 4 axis weightings = 20 scenarios:

| Model | wins | worst position |
|---|---|---|
| GLM 5.3 Flash | **13** | #3 |
| Gemini 3.8 Flash | 5 | #4 |
| DeepSeek V4.1 Flash | 2 | #4 |
| GLM-5.3 | 0 | #5 |
| GPT-5.6 Luna | 0 | #5 |

GLM 5.3 Flash wins 13 of 20 and is never worse than #3, while GLM-5.3 — which tops the plain
composite — wins **zero**. But widen to all 72 comparable models and sweep 3 mixes × 4
weightings = 12 scenarios, and **no model at all stays in the top 6.** Only two stay in the
top 10. The spread is brutal: `gpt-6-astra` ranges from #11 to #63 depending on the scenario.

**So:** report the worst case, not the best. A model that wins one emphasis is a bet on that
emphasis. And "no model survives the sweep" is a legitimate, useful finding — it means the
question was underspecified, not that the analysis failed.

## 11. Some headline speeds are still moving

A speed figure is a snapshot. Seven daily points per model:

| Model | 7-day median | range | swing | drift |
|---|---|---|---|---|
| GLM 5.3 Flash | 99.3 | 56.4–126.6 | **70.7%** | −51.4% |
| GPT-5.6 Luna | 145.3 | 115.9–183.3 | 46.4% | +20.9% |
| GLM-5.3 | 56.7 | 50.8–76.6 | 45.4% | −2.5% |
| Gemini 3.8 Flash | 320.6 | 260.4–383.6 | 38.4% | −21.7% |
| DeepSeek V4.1 Flash | 217.2 | 197.4–242.4 | 20.7% | +3.9% |

GLM 5.3 Flash nearly halved in a week. The catalogue's single speed number for it is a
measurement of one day.

**So:** before ranking on speed, check the drift — and note the route reports `planLimitDays: 7`,
so **one week is the ceiling**. This is a stability check, never a history source.

---

## Reproducing these numbers

```bash
python3 -m unittest test_aa_fetch.py          # 40 offline tests, no network
python3 aa_fetch.py rank --json               # the 72 comparable models, tie-merged
python3 aa_fetch.py compare <a> <b> --scores  # per-eval scores + interval verdict
python3 aa_fetch.py compare <a> <b> --stability  # the 7-day drift in finding #11
```

**Checked, and it did not matter.** Speed is measured at four prompt lengths (`medium`,
`long`, `hundredK`, `mediumParallel`) and the ordering is **identical at all four** — so
quoting a speed at the default is safe. Worth writing down so nobody re-derives it.

Two cautions:

- **Everything here drifts, not just the counts.** Taken 2026-09-22 against a live catalogue.
  The counts (656 / 381 / 496) move by a few models, and the individual figures move too —
  speeds especially, which is finding #11. Re-run the commands rather than quoting these.
  The proportions are the finding; every number is one measurement.
- **Tie counts depend on how you count.** "43 of 72 share a tied quality rank" counts models
  with a fractional rank; `rank` prints 41 rows with a composite score, because two are
  missing another axis. Same data, two defensible conditions — worth knowing before you
  conclude one of us is wrong.

Findings #1 and #9 both rest on `briefcaseBreakdown.overall`, the single evaluation that
publishes an interval. If the site stops publishing it, neither can be checked at all.

## What to take away

1. **Check error bars before believing a ranking.** Where they overlap, say "tied" and mean it.
2. **Never read a conditional metric without its base rate.** Hallucination rate needs accuracy.
3. **Compute cost at your own mix.** A blended price column is someone else's workload.
4. **Filter deprecated and estimated entries first.** 58% and 76% is most of the table.
5. **Decompose the composite and reweight it.** The leader wins on someone else's workload.
6. **Sweep every assumption, and report the worst case.** Vary the weights *and* the mix — and
   "no model survives" is a real answer.
7. **Check whether a zero means zero or "not measured".** Those are different claims.
8. **Prefer a gap to a guess** when data is missing — in your analysis and in your output.

None of this is a criticism of the people publishing these benchmarks. The numbers are fine,
the provenance is unusually well documented, and the confidence intervals that made finding
#1 possible are published rather than hidden. The problems are all in what survives the trip
from "field in a dataset" to "row in a leaderboard".
