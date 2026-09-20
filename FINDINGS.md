# Six things LLM leaderboards hide

I spent a day pulling LLM benchmark data apart programmatically, across 653 models and
~89 fields each. These are the findings that survived checking. The theme: **every
individual number can be correct while the conclusion you draw from it is wrong.**

All figures below are from Artificial Analysis's public data as of 2026-09-20. The tool
behind them is [in this repo](README.md).

---

## 1. Nearly half the leaderboard is noise

Leaderboards rank models in a strict order. Most of those adjacent pairs are not actually
separable.

The catalogue publishes a 95% confidence interval for one evaluation (aa-briefcase). Across
the 69 models with complete enough data to compare, **31 of the 68 adjacent pairs have
overlapping intervals** — statistically indistinguishable.

```
claude-fable-5-1 (1678)  and  claude-opus-5 (1673)   overlap
glm-5-3 (1525)           and  grok-4-6 (1523)        overlap
grok-4-6 (1523)          and  kimi-k3 (1511)         overlap
```

When intervals are merged rather than ranked, **35 of those 69 models share a tied quality
rank** — 33 of them also carry a full composite score; the other two are missing a different
axis entirely, which is finding #6 in miniature.

This is the most important thing here, and it's invisible in every UI that shows a numbered
list. A model at #3 and a model at #4 may be the same model as far as the evidence goes. The
numbers aren't wrong; the *ordering* is an artifact.

**Practical consequence:** if you're choosing between two models that are a few places apart,
check the intervals before paying attention to the rank. A 65-point gap with disjoint
intervals is a real difference. A 2-point gap with overlapping intervals is a coin flip that
the leaderboard will happily present as #1 vs #2.

---

## 2. The hallucination rate is conditional — and read alone it inverts

There's a well-known "hallucination rate" figure. Read it as "share of answers that are made
up" and you will get the ranking exactly backwards.

It's the share of **wrong** answers that were confabulations rather than abstentions. A model
that says "I don't know" constantly has a beautiful hallucination rate and terrible accuracy.

| Model | accuracy | hallucination rate | actual confabulations per 100 questions |
|---|---|---|---|
| GLM-5.3 | 0.339 | 29.6% | **19.6** |
| GLM 5.3 Flash | 0.275 | 27.6% | **20.0** |
| Gemini 3.8 Flash | 0.546 | 55.2% | 25.1 |
| DeepSeek V4.1 Flash | 0.464 | 96.5% | 51.7 |
| GPT-5.6 Luna | 0.427 | 92.6% | 53.1 |
| `command-a-plus` | 0.089 | **14.2%** | 12.9 |

`command-a-plus` has the *lowest* hallucination rate in the set — because it answers wrong
by abstaining, at 8.9% accuracy. It isn't honest; it's quiet.

And note the ranking flip: by raw rate, GLM 5.3 Flash looks worst of the well-known models
at 27.6%. By actual confabulations it's roughly **2.5× better** than Luna or DeepSeek.

**Practical consequence:** the absolute figure is `(1 − accuracy) × hallucination_rate`.
Never read the rate without the accuracy next to it.

---

## 3. Price rankings reverse depending on your workload

"Price per million tokens" is one number covering a range of very different bills, because
input, output and cached input cost wildly different amounts and the mix depends entirely on
what you're doing.

Two models can swap places as you change the mix. For one pair I checked, the crossover sits
at roughly a **93% cache-hit rate** — above it one model is cheaper, below it the other.

The default blend most sites show (often something like 0:3:1 cached:input:output) is a
reasonable average and a bad fit for any specific workload. A long-document summarizer and a
chatbot have completely different mixes.

**Practical consequence:** compute the price at your own mix before believing a cost ranking.
If you don't know your cache-hit rate, you don't know which model is cheaper.

---

## 4. 58% of the catalogue is deprecated, and 76% of the headline scores are estimates

- **381 of 653 models are deprecated.** Most carry a pointer to their replacement, which is
  genuinely useful — but nothing flags it at a glance.
- **497 of 653 intelligence indices are estimated**, not measured. The estimate is usually
  labelled in the data, but a table of scores doesn't show you which rows are which.

**Practical consequence:** a large fraction of any "compare all models" exercise is
comparing things that are retired or guessed at.

---

## 5. Some latency figures are meaningless

Latency is split into input / reasoning / answer. Some reasoning models report **zero
reasoning time** — the thinking is folded into the input bucket instead.

| Model | total | input | reasoning | answer |
|---|---|---|---|---|
| GLM-5.3 | 37.66 | 2.99 | 27.73 | 6.93 |
| Gemini 3.8 Flash | 18.07 | **16.44** | **0.00** | 1.64 |
| GPT-5.6 Luna | 126.37 | **123.28** | **0.00** | 3.09 |

That 123-second "time to first chunk" isn't latency. It's untimed reasoning sitting in the
wrong column. Any comparison that ranks these models on responsiveness is comparing a
measurement against a non-measurement.

**Practical consequence:** if a reasoning model reports 0s of reasoning time, its latency
figures aren't comparable to models that report it.

---

## 6. Missing data gets scored as good data — a bug I wrote myself

This one is a cautionary tale rather than a data finding.

I built a ranking that combined several axes (quality, cost, speed, stability) into one
score by averaging. Models with no published price have no cost axis — so they were averaged
over the axes they *did* have. A model measured on one axis was compared against models
measured on three.

A quality-only model ranked #1 would have scored 1.00 and appeared to beat a model that was
top-3 on everything.

The fix was to refuse to score such models at all, and say why. That's a general rule worth
more than the specific bug: **when data is missing, the honest output is a gap, not a
number.** Averaging over what you have rewards the absence.

It survived every test I'd written, because all my tests used models with complete data.
Edge cases with *missing fields* are a different test class from malformed input, and I'd
only written the second kind.

---

## Reproducing these numbers

Every figure here comes from one source and can be re-derived without trusting me:

```bash
python3 -m unittest test_aa_fetch.py         # 39 offline tests, no network needed
python3 aa_fetch.py rank --json              # the 69 comparable models, tie-merged
python3 aa_fetch.py compare <a> <b> --scores # per-eval scores + interval verdict
```

Two cautions if you do:

- **The counts drift.** They were taken on 2026-09-20 and the catalogue is live, so the
  653 / 381 / 497 numbers move by a few models over time.
- **Tie counts depend on how you count.** "35 of 69 share a tied quality rank" counts models
  with a fractional rank; `rank` prints 33 rows with a composite score, because two are
  missing a different axis. Same data, two defensible conditions — worth knowing before you
  conclude one of us is wrong.

Both headline findings come from `briefcaseBreakdown.overall`, the single evaluation that
publishes a confidence interval. If the site stops publishing it, findings #1 and #6 can no
longer be checked at all.

## What I'd actually take away

1. **Check error bars before believing a ranking.** Where they overlap, say "tied" and mean it.
2. **Never read a conditional metric without its base rate.** Hallucination rate needs accuracy.
3. **Compute cost at your own mix.** A blended price column is someone else's workload.
4. **Filter deprecated and estimated entries before comparing.** 58% and 76% is most of the table.
5. **Check whether a zero means zero or means "not measured."** Those are different claims.
6. **Prefer a gap to a guess** when data is missing — in your analysis and in your output.

None of this is a criticism of the people publishing these benchmarks. The individual
numbers are fine, the provenance is unusually well documented, and the confidence intervals
that made finding #1 possible are published rather than hidden. The problems are all in what
survives the trip from "field in a dataset" to "row in a leaderboard".
