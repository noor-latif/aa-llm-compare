# aa-llm-compare

Programmatic LLM comparison from [Artificial Analysis](https://artificialanalysis.ai) —
standard library only, no dependencies, no API key.

**Unofficial research client.** It reads the figures the public site renders, through the
site's own internal routes. Not affiliated; those routes are undocumented and may change or
gain auth without notice, and **for anything you depend on in production or commercially you
should use their official API.** One cached fetch, never polling.

> **Just want the findings?** → **[What LLM leaderboards hide](FINDINGS.md)** — no code:
> why 31 of 68 adjacent leaderboard pairs are statistically indistinguishable, why the
> hallucination rate inverts when read alone, and the rest.

```bash
python3 aa_fetch.py compare glm-5-3 glm-5-3-flash gemini-3-8-flash --evals --stability
```

```
slug                   effort     II      $/M   tok/s p05-p95       hosts
-------------------------------------------------------------------------
glm-5-3                max     44.78     2.150    61.0  30-94           22
glm-5-3-flash          max     41.81     0.237    89.4  46-140          19
gemini-3-8-flash       high    40.93     1.500   343.5 230-456           4

$/M at 0:3:1 (cached:in:out). ~ = estimated index.
WARN: mixed effort levels ['high', 'max'] -- not apples-to-apples
```

The fetching is the easy part. The value is the **warnings** — this dataset is full of
numbers that look comparable and aren't. [FINDINGS.md](FINDINGS.md) explains each one.

## Requirements

Python 3.9+ (CI-verified on 3.9–3.13). No third-party packages. No API key.

## Quickstart

```bash
git clone <this repo> && cd aa-llm-compare
python3 -m unittest test_aa_fetch.py   # 40 offline tests, no network, <1ms
python3 aa_fetch.py demo               # canary: hits the live site, asserts it still behaves
python3 aa_fetch.py rank               # the ~72 comparable models, in one ordering
```

`demo` is the tripwire. It asserts the catalogue shape, that `?compare=` is still ignored,
that `blended()` reproduces every published price ratio, and that each guard actually fires.
**Run it after the site changes** — if AA rebuilds, that is where it shows.

## Commands

| Command | What it does |
|---|---|
| `demo` | Self-check against the live site |
| `catalogue <file.json>` | Dump the whole catalogue, all fields |
| `compare <slug>...` | Side-by-side table + warnings |
| `trustworthy` | The ~72 models that can actually be compared |
| `rank` | Collapse quality/cost/speed/stability into one ordering |

Flags work on `compare`, `trustworthy` and `rank`:

| Flag | Meaning |
|---|---|
| `--mix 0:3:1` | Price mix as `cached:input:output` (default `0:3:1`) |
| `--scores` | Every benchmark score side by side, plus the fields with no data |
| `--evals` | Rank within each of the 10 evals behind the index |
| `--weights k=2,...` | Weight those evals, e.g. `--weights terminalbench-4-0=3` |
| `--stability` | 7-day speed swing and drift |
| `--rank` | Collapse every axis into a single ordering |
| `--json` | Machine-readable output |

Ten warnings fire automatically on `compare`. **Three do not**, because acting on them
costs an extra request or only means something for your specific workload — the price mix
(`--mix`), composite blind spots (`--scores` / `--evals` / `--weights`), and speed drift
(`--stability`). Treat the automatic ones as the floor, not the ceiling.

The table shows `slug`, `effort`, index, `$/M` at your mix, `tok/s` with its `p05-p95`
spread, `hosts`, `halluc%`, `acc`, `context`. `--json` exposes more than fits: `params`,
`activeParams` (MoE-active), `license`, `elo` with its interval, `suiteTokens` (total output
tokens for the whole benchmark suite), and `supersededBy` — the slug that replaced a
deprecated model, so the warning reads `deprecated -- superseded by X`.

Library use:

```python
import aa_fetch

models = aa_fetch.catalogue()                           # the whole catalogue
aa_fetch.blended(models[0], cached=100, inp=1, out=1)   # price at any workload mix
aa_fetch.compare(["glm-5-3", "gemini-3-8-flash"])       # (rows, warnings)
aa_fetch.trustworthy()                                  # ~72 comparable models
aa_fetch.rank(["glm-5-3", "glm-5-3-flash"], axis_weights={"cost": 3})
```

## The endpoint

```
GET /models/<any-slug>      + header "RSC: 1"
```

No auth. ~2.9 MB of `text/x-component` containing the full model catalogue. It is a Next.js
App Router page: the whole catalogue ships inside the React Server Component flight payload
and the browser filters it locally, so there is no API call to intercept and `?compare=` is
ignored by the server (four different compare sets returned byte-identical responses). Any
slug works — the catalogue section is identical regardless.

Two parsing gotchas. **Creator objects** (`zai`, `openai`, `anthropic`) match the same
`{"id":"<uuid>","slug":"` shape as models, and the payload contains **twice as many
matching objects as there are models** — every model, plus a creator object. The
discriminator is the presence of a `creator` key; without it you silently ingest hundreds of
junk rows, and the ratio is not stable (it has been ~59 creators and ~656), so never hardcode
it. And `timescaleData` is two floats, not a time series.

Also available: `/api/models/performance-over-time` (POST, no auth, takes UUIDs not slugs,
7 daily points — the ceiling) and `/api/v2/language/models` (GET, `x-api-key`, the official
supported API).

## Responsible use

These are **undocumented internal routes**, not a supported API.

- **Cache.** One catalogue request is 2.9 MB and the edge caches it for hours. `catalogue()`
  is `lru_cache`d, so multiple operations share one fetch.
- **Don't poll.** The data changes daily at most.
- **Use the official API** for anything production-facing or commercial.
- All figures here are Artificial Analysis's; this repo is an unaffiliated client.

## License

MIT — see [LICENSE](LICENSE).

Findings dated 2026-09-22 against a 656-model catalogue. The counts drift as the catalogue
changes; the properties behind them are structural.
