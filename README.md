# aa-llm-compare

Programmatic LLM comparison from [Artificial Analysis](https://artificialanalysis.ai) —
standard library only, no dependencies.

**Unofficial research client.** It reads the same figures the public site renders, through
the site's own internal routes. It is not affiliated with Artificial Analysis, those routes
are undocumented and may change or gain auth without notice, and **for anything you depend
on in production or commercially you should use their official API.** One cached fetch, never
polling — see [Responsible use](#responsible-use).

```bash
python3 aa_fetch.py compare glm-5-3 glm-5-3-flash gemini-3-8-flash --evals --stability
```

```
slug                   effort     II      $/M   tok/s p05-p95       hosts
-------------------------------------------------------------------------
glm-5-3                max     44.78     2.150    72.1  38-118          22
glm-5-3-flash          max     41.81     0.237    95.2  69-141          18
gemini-3-8-flash       high    40.93     1.500   305.0 233-482           4

$/M at 0:3:1 (cached:in:out). ~ = estimated index.
WARN: mixed effort levels ['high', 'max'] -- not apples-to-apples
```

Most of the value here is not the fetching. It is the **warnings**: this dataset is full
of numbers that look comparable and aren't. See [How this data misleads](#how-this-data-misleads).

---

## Why this exists

If you open the comparison page and look at the Network tab, you will find **nothing to
intercept**. It is a Next.js App Router page: the entire model catalogue ships inside the
React Server Component (RSC) flight payload, and the browser filters it locally.

Two consequences that surprised us:

1. **The `?compare=` parameter is a lie.** The server ignores it. We fetched four different
   compare sets — including no parameter at all — and got byte-identical responses
   (557,675 bytes every time). You cannot use it to reduce payload size.
2. **There is no API call to reverse-engineer** — so ask for the flight payload instead:

```
GET /models/<any-slug>      + header "RSC: 1"
```

Returns `text/x-component`, ~2.9 MB, **no auth**, containing **653 full model objects**
with 89 fields each. Any slug works: the catalogue section is identical regardless
(verified across two anchors, zero differing objects) and stable on re-fetch.

## Requirements

Python 3.9+ (CI-verified on 3.9, 3.10, 3.11, 3.12, 3.13). **No third-party packages.** No API key.

## Quickstart

```bash
git clone <this repo> && cd aa-llm-compare
python3 -m unittest test_aa_fetch.py  # offline logic tests, no network, <1s
python3 aa_fetch.py demo        # self-check; hits the live site, asserts everything still works
python3 aa_fetch.py rank        # the ~69 models worth comparing, in one ordering
```

`test_aa_fetch.py` covers the pure logic (blended price math, tie merging, CLI
argument parsing) without the network — 19 tests, runs in milliseconds. `demo` is the
canary. It asserts the catalogue shape, that `?compare=` is still ignored, that `blended()`
reproduces every published price ratio to 1e-9, and that each data-quality guard
actually fires. **Run both after the site changes.**

## Commands

| Command | What it does |
|---|---|
| `demo` | Self-check against the live site |
| `catalogue <file.json>` | Dump all 653 models, all fields |
| `compare <slug>...` | Side-by-side table + warnings |
| `trustworthy` | The ~69 models that can actually be compared |
| `rank` | Collapse quality/cost/speed/stability into one ordering |

Flags work on `compare`, `trustworthy` and `rank`:

| Flag | Meaning |
|---|---|
| `--mix 0:3:1` | Price mix as `cached:input:output` (default `0:3:1`) |
| `--evals` | Rank models within each of the 10 evals behind the index |
| `--scores` | Every benchmark score side by side: the 10 composite evals **plus** all standalone fields with data, and an explicit list of the ones with none. Ranks hide gap sizes (1526 vs 1461 and 1526 vs 1525 both render as "#1 vs #2") — use this when you want magnitudes. |
| `--weights k=2,...` | Weight those evals, e.g. `--weights terminalbench-4-0=3` |
| `--stability` | 7-day speed swing and drift |
| `--rank` | Collapse every axis into a single ordering |
| `--json` | Machine-readable output (redirect to a file to save a leaderboard) |

The printed table shows: `slug`, `effort`, intelligence index, `$/M` at your mix, `tok/s`,
its `p05-p95` spread, `hosts`, `halluc%`, `acc`, and `context` (in thousands, to match the
source's own display — which is what makes Kimi's 1049k visible against everyone else's
1000k). `--json` exposes more than the table can fit, including `params` (total
parameters), `activeParams` (active, i.e. MoE), `license`, `elo` with its interval,
`context` in raw tokens, `suiteTokens` — the total output tokens consumed running the whole
benchmark suite, which is what the "cost to run the intelligence index" number is actually
buying (DeepSeek: 253M output tokens for $477) — and `supersededBy`, the slug of whatever
replaced a deprecated model. 375 of the 381 deprecated models name their own successor, so
the warning reads `deprecated -- superseded by X` instead of stopping at "deprecated".

One caveat on `context`: it reports `contextWindowTokens` exactly as served. For DeepSeek
V4.1 Flash the source page shows a **different** figure — `~1,500k (estimated)` against the
served `1,000k` — and DeepSeek is the only model on that page whose context is marked
"(estimated)". No 1.5M value exists anywhere in the payload, so that figure is AA's own
estimate rather than a field they publish. Where the source marks a value as estimated, the
served field is the official spec and the page's number is their judgement; read them as two
different claims rather than a mismatch. This is the second such opacity found — the first
was the headline blended price, which uses an undisclosed default mix.

Library use:

```python
import aa_fetch

models = aa_fetch.catalogue()                       # 653 dicts, 89 fields each
by_slug = {m["slug"]: m for m in models}

aa_fetch.blended(models[0], cached=100, inp=1, out=1)   # price at any workload mix
aa_fetch.compare(["glm-5-3", "gemini-3-8-flash"])       # (rows, warnings)
aa_fetch.trustworthy()                                  # ~69 comparable models
aa_fetch.composite(["glm-5-3", "gpt-5-6-luna"], {"gdp-pdf": 3})
aa_fetch.rank(["glm-5-3", "glm-5-3-flash"], axis_weights={"cost": 3})
```

---

## How this data misleads

Nine of these twelve fire as warnings the moment you run `compare`. **Three do not** —
they are opt-in, because acting on them either costs an extra request or only means
something for your specific workload. No warning will fire for these; you have to ask:

- **#4 price mix** — use `--mix`. The footer always states the mix in use (`$/M at
  0:3:1`…), so the default is visible, but nothing warns you it might be wrong for you.
- **#7 composite blind spots** — use `--evals` (ranks), `--scores` (raw scores), or
  `--weights` (your own emphasis). A model can lead the composite and lose your workload.
- **#8 speed drift** — use `--stability`. It costs one POST per model, so it is not run by
  default; a speed figure shown without it may be a week out of date.

Treat the nine automatic warnings as the floor, not the ceiling.

Every one of these is guarded in code and asserted in `demo()`. They are the reason this
repo exists — a comparison built straight off the raw fields will be wrong.

**1. 76% of intelligence indices are estimated, not measured.** `intelligenceIndexIsEstimated`
is true for **497 of 653** models. `performanceDataSource.type` is `firstParty` (304,
provider-reported) or `median` (349 — imputed from the catalogue median, i.e. a guess).
`compare()` marks these with `~`.

**2. 58% of the catalogue is deprecated.** 381 of 653. Nothing flags this at a glance.

**3. Effort level is a hidden axis.** `effort.label` is `low|medium|high|xhigh|max`, and one
model ships as several slugs — `gemini-3-8-flash` exists as `-low` (33.45), `-medium` (39.77)
and `high` (40.93). Comparing `max` against `high` is not apples-to-apples, and ranking the
raw catalogue counts one model four times. `trustworthy()` collapses these.

**4. Price depends entirely on your workload.** Don't trust a single blended price column.
The published `price1mBlended*` fields are `cached:input:output` ratios — verified, e.g.
`0To3To1 == (3*input + 1*output)/4` — and **the ranking reverses**:

| Mix | Cheapest of the five |
|---|---|
| `0:3:1` (balanced) | GLM 5.3 Flash, $0.238 |
| `100:1:1` (cache-heavy agent loop) | **DeepSeek V4.1 Flash, $0.021** |

`blended()` derives any mix from the component prices. Better still, **solve for the
crossover instead of guessing your mix**: DeepSeek's cache price ($0.006/M) undercuts
GLM 5.3 Flash's ($0.026/M) 4.3×, and at input:output 3:1 the two tie at **57.5 cached
tokens per 4 fresh — a 93.5% cache-hit rate**. So "DeepSeek is cheapest at 100:1:1" is
technically true and practically irrelevant. GLM 5.3 Flash wins at every realistic mix.

**5. A median speed without its spread is misleading.** `outputSpeedVariance` carries
`p05/q25/median/q75/p95`, and `hostModelCount` says how many providers back it. Gemini's
305 tok/s spans 233–482 on **4 hosts**; GLM-5.3's 72.1 spans 38–118 on **22 hosts**. The
faster number is the less reliable one.

**6. Some comparable models have no price at all.** Six of the trustworthy set are
open-weights with zero hosts and no published API price. They pass every quality filter and
still cannot be cost-ranked. `compare()` renders `n/a` and warns.

**7. The composite index hides opposite strengths.** `intelligenceIndexEvaluations` carries
the 10 evals behind the index. `--evals` ranks within each — not against raw scores, which
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

GLM-5.3 tops the composite (44.78) and wins 4/10, but is **dead last on long-context
reasoning and gdp-pdf**. If that is your workload, the composite ranking is actively wrong.

**8. Some headline speeds are still moving.** `--stability` pulls 7 daily points per model:

| Model | 7-day median | range | swing | drift | |
|---|---|---|---|---|---|
| GPT-5.6 Luna | 120.2 | 108.0–183.3 | **62.7%** | **+69.7%** | volatile, trending up |
| Gemini 3.8 Flash | 320.6 | 260.4–383.6 | 38.4% | +44.5% | trending up |
| GLM 5.3 Flash | 106.1 | 92.0–126.6 | 32.6% | −5.3% | stable |
| GLM-5.3 | 71.6 | 54.8–76.6 | 30.5% | −0.6% | stable |
| DeepSeek V4.1 Flash | 216.9 | 197.4–247.0 | **22.9%** | −4.8% | most stable |

Luna went from 108 to 183 tok/s in seven days, so its catalogued 161.8 is a moving target.
Note the anonymous time-series route reports `planLimitDays: 7` — **one week is the ceiling**,
so this is a stability check, never a history source.

**9. Weighting the evals changes the winner — including to the worst-ranked model.**
`composite()` min-max normalises each eval across the compared set, then takes a weighted mean:

| Emphasis | Winner |
|---|---|
| equal weights | GLM-5.3 |
| coding (`terminalbench-4-0`:3, `scicode`:2, `critpt`:1) | GLM-5.3 |
| knowledge (`humanitys-last-exam`:3, `omniscience`:3) | **Gemini 3.8 Flash** |
| long-context (`aa-long-context-reasoning`:3, `gdp-pdf`:1) | **GPT-5.6 Luna** |
| enterprise (`gdp-pdf`:3, `gdpval-aa`:2, `aa-briefcase`:2) | **GLM 5.3 Flash** |

GPT-5.6 Luna is **last** on the composite (37.32) and **first** under long-context
weighting. GLM 5.3 Flash wins **zero** individual evals yet wins the enterprise weighting —
it is #2 or #3 on all three. Normalising rewards consistency over spiky excellence, which is
the opposite of what a wins-count tells you. Neither view is wrong; they answer different
questions.

Caveats on `composite()`: min-max normalisation is **relative to the compared set** — add a
model and every score moves, and the worst model in each eval always scores 0 (coarse at
n=5). A model with no score for a weighted eval counts as worst.

**10. Hallucination is buried, and the rate is conditional.** There is no top-level
`hallucination` field anywhere — it lives inside `omniscienceBreakdown` as
`{accuracy, hallucinationRate}`, present for 528 of 653 models and all 69 of the
trustworthy set. `compare()` surfaces both as `halluc%` and `acc`.

The rate is the share of **wrong** answers that are confabulations rather than abstentions
— not a share of all answers. Read alone it misleads in both directions:
`command-a-plus` has the lowest rate in the trustworthy set (14.2%) but only 8.9% accuracy.
It abstains constantly; that is caution, not honesty. Absolute confabulation is
`(1 - accuracy) × hallucinationRate`:

| Model | accuracy | hallucination rate | confabulations per 100 questions |
|---|---|---|---|
| GLM-5.3 | 0.339 | 29.6% | **19.6** |
| GLM 5.3 Flash | 0.275 | 27.6% | **20.0** |
| Gemini 3.8 Flash | 0.546 | 55.2% | 25.1 |
| DeepSeek V4.1 Flash | 0.464 | 96.5% | 51.7 |
| GPT-5.6 Luna | 0.427 | 92.6% | 53.1 |
| `command-a-plus` | 0.089 | 14.2% | 12.9 |

So on this axis the two GLM models are roughly 2.5x better than Luna and DeepSeek — and
the raw rate ranking (which put GLM last at 27.6%) would have told you the opposite if you
read it as "share of answers that are hallucinations."

**11. Nearly half the leaderboard is noise.** `briefcaseBreakdown.overall` is the only
place AA publishes an interval (`elo` with `lower95ci` / `upper95ci`). Across the 69
trustworthy models, **31 of the 68 adjacent pairs have overlapping 95% intervals** — they
are statistically indistinguishable. `claude-fable-5-1` (1678) and `claude-opus-5` (1673)
overlap. So does `glm-5-3` (1525) and `grok-4-6` (1523). `compare()` warns on every
overlapping adjacent pair. Ordering models that the data cannot separate is the most common
way a leaderboard lies while every individual number stays true.

**12. Reasoning time is not always timed.** `endToEndResponseTime` splits latency into
`input` / `reasoning` / `answer`. Some reasoning models report `reasoning: 0.00` — the
thinking time is silently folded into `input` instead:

| Model | total | input | reasoning | answer |
|---|---|---|---|---|
| GLM-5.3 | 37.66 | 2.99 | 27.73 | 6.93 |
| DeepSeek V4.1 Flash | 13.20 | 1.15 | 9.64 | 2.41 |
| Gemini 3.8 Flash | 18.07 | **16.44** | **0.00** | 1.64 |
| GPT-5.6 Luna | 126.37 | **123.28** | **0.00** | 3.09 |

That is the explanation for the absurd 123.28s time-to-first-chunk noted earlier: it is
not latency, it is un-timed reasoning. `compare()` flags any reasoning model reporting 0s
and tells you not to compare its latency. Other genuinely nested metrics worth knowing
about: `capabilities.{engineering,legal,economics,financeAndAccounting,strategyAndOps,
healthcareAndMedical}` (domain scores, ~150 models) and `openness.opennessIndex` (322).

## Collapsing to one ordering

`rank()` averages **per-axis ranks** across quality / cost / speed / stability — ranks, not
scores, because $/M, tok/s and the index are incomparable units. Equal weight is the default
and is a stated assumption, not a finding; pass `axis_weights={"cost": 3}` to shift it.

| # | Model | score | quality | cost | speed | stability |
|---|---|---|---|---|---|---|
| 1= | GLM 5.3 Flash | 2.50 | #2 | **#1** | #4 | #3 |
| 1= | DeepSeek V4.1 Flash | 2.50 | #4 | #3 | #2 | **#1** |
| 3 | Gemini 3.8 Flash | 3.00 | #3 | #4 | **#1** | #4 |
| 4 | GLM-5.3 | 3.25 | **#1** | #5 | #5 | #2 |
| 5 | GPT-5.6 Luna | 3.75 | #5 | #2 | #3 | #5 |

The top two tie *in this one scenario*, and that is worth stressing: the tie is
scenario-specific. Sweeping 5 price mixes × 4 axis weightings = 20 scenarios,
`glm-5-3-flash` takes #1 in **12 of 20 and is never worse than #3**, while
`deepseek-v4-1-flash` wins 3 and `gemini-3-8-flash` wins 4 (all four of those under
speed-heavy weighting). At the cache-heavy `100:1:1` mix, DeepSeek leads on three of the
four weightings — the crossover described in trap #4 showing up again. So among these five,
GLM 5.3 Flash is the robust pick rather than half of a tie; `glm-5-3` wins once and is #5
at worst, and `gpt-5-6-luna` never wins at all.

Still different bets underneath: GLM 5.3 Flash wins on price and quality, DeepSeek on speed
and stability. Test robustness across **every dimension that moves the answer**, not just the obvious one.
Ranking all 69 trustworthy models under four axis weightings (equal / cost 3x / quality 3x /
speed 3x) at one price mix suggests `ling-3-0-flash-fin` is top-6 everywhere. Add the price
mix as a second dimension — three mixes (0:3:1, 100:1:1, 0:100:1) × four weightings = 12
scenarios — and **no model at all stays in the top 6**:

| Model | worst case | best case |
|---|---|---|
| `ling-3-0-flash-fin` | **#7** | #1 |
| `deepseek-v4-1-flash` | #8 | #2 |
| `ling-3-0-flash-vl` | #9 | #1 |
| `k2-horizon-375b-a23b` | #14 | #8 |
| `deepseek-v4-flash-vision` | #15 | #1 |

Only **three models stay inside the top 10 across all 12** (`ling-3-0-flash-fin`,
`deepseek-v4-1-flash`, `ling-3-0-flash-vl`), and five inside the top 15. Contrast
`claude-fable-5-1`: quality #3, but cost #51 and speed #39 — the most extreme specialist in
the set, and a bad general default.

This is the second correction to this section. Tie-merging moved the "most robust" answer
once; varying the price mix removed the top-6 claim entirely. Both times the underlying
numbers were correct and only the scope of the claim was wrong.

Overlapping intervals are merged on the quality axis, so **35 of the 69 share a tied
quality rank** (scored as the average of the positions they span). That is why these
figures differ from a strict ranking, and it **corrected an earlier claim in this README**:
before ties were merged, `deepseek-v4-1-flash` looked like the top-6-everywhere pick. Its
worst case is actually #8. Merging the ties moved the answer.

## Checked, and it did not matter

Recorded so nobody redoes it. Every claim above survived a sweep; these did not change
anything:

- **Prompt length.** Speed is measured at four prompt types (`medium`, `long`, `hundredK`,
  `mediumParallel`). Across the five reference models the speed ordering is **identical at
  all four** (Gemini > DeepSeek > Luna > GLM 5.3 Flash > GLM-5.3), so `compare()`'s default
  of `long` is safe. Absolute values do move — Luna ranges 103–162 tok/s across prompt
  types — which is why `--stability` exists.
- **Ties among the five.** None of the five reference models has overlapping intervals with
  another, so their ordering is not affected by tie merging. Among the full 69 it matters a
  great deal (35 of 69 tied).

## The endpoints

| Route | Method | Auth | Notes |
|---|---|---|---|
| `/models/<slug>` + `RSC: 1` | GET | none | 2.9 MB, 653 models, 89 fields. **The one that matters.** |
| `/models/comparisons` + `RSC: 1` | GET | none | 557 KB, 272 summary entries. Strict subset of the above — dropped from the code. |
| `/api/models/performance-over-time` | POST | none | 7 daily points per model. Body: `{endpoint, modelIds[], hostModelIds[], promptType}` where `promptType` is `medium\|long\|100k`. Takes UUIDs, not slugs. |
| `/api/v2/language/models` | GET | `x-api-key` | The official, supported, paid API. Use this in production. |

Parsing gotcha: **creator objects** (`zai`, `openai`, `anthropic`) match the same
`{"id":"<uuid>","slug":` shape as models. 712 objects match on a per-model page but only 653
are models — the other 59 are creators. The discriminator is the `creator` key: models have
one, creators don't. Without it you silently ingest 59 junk rows.

Also worth knowing: `timescaleData` is not a time series despite the name — it is two floats.

If the site is rebuilt, the two things most likely to break are the `_MODEL_START` regex
(key ordering) and the `RSC: 1` contract. Both are asserted in `demo()`.

## Responsible use

These are **undocumented internal routes**, not a supported API. They can change or gain
auth without notice.

- **Cache aggressively.** One catalogue request is 2.9 MB and the site edge-caches it for
  hours; re-fetching per comparison is rude and pointless. `catalogue()` is `lru_cache`d so
  multiple operations share one fetch.
- **Don't poll.** The data changes daily at most.
- **Use the official API** (`/api/v2/language/models`) for anything production-facing or
  commercial. It is keyed, supported, and the right way to depend on this data.
- All figures here are Artificial Analysis's; this repo is an unaffiliated client.

## License

MIT — see [LICENSE](LICENSE).

Findings dated 2026-09-20 against a 653-model catalogue. Counts (653 / 497 / 381 / 104 / 69)
will drift as the catalogue changes; the properties behind them are structural.
