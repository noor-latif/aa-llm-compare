#!/usr/bin/env python3
"""Compare LLM models using artificialanalysis.ai data. Stdlib only, no API key.

The site has no client-side API to intercept: it is a Next.js App Router page that ships
the whole model catalogue inside the React Server Component flight payload. Asking for
that payload directly (header `RSC: 1`) returns the whole model catalogue, unauthenticated.

  catalogue()   -> the full model catalogue      series()    -> 7-day speed history
  compare()     -> a comparison that flags the ways the raw numbers mislead
  trustworthy() -> the ~69 models worth comparing, effort variants collapsed
  composite()   -> weighted score over the 10 evals behind the index
  rank()        -> quality/cost/speed/stability collapsed into one ordering

  python3 aa_fetch.py demo                          # self-check, hits the live site
  python3 aa_fetch.py catalogue catalogue.json      # every model, all fields
  python3 aa_fetch.py compare <slug>... [flags]     # side-by-side + warnings
  python3 aa_fetch.py trustworthy [flags]           # the ~69, ranked
  python3 aa_fetch.py rank [flags]                  # one ordering to rule them all

  --mix 0:3:1        cached:input:output mix for price (default 0:3:1)
  --evals            rank the models within each of the 10 composite evals
  --weights k=2,...  weight those evals, e.g. --weights terminalbench-4-0=3
  --stability        7-day speed swing/drift from the time-series route
  --rank             collapse every axis into one ordering
  --json             machine-readable rows + warnings; redirect to a file to save

Findings, caveats and the endpoint reference are in README.md.
"""

import difflib
import functools
import json
import re
import statistics
import sys
import urllib.request

BASE = "https://artificialanalysis.ai"
UA = "Mozilla/5.0 (compatible; research-script/1.0)"

# A model detail object always starts with an id/slug pair in this exact order.
_MODEL_START = re.compile(r'\{"id":"[0-9a-fA-F-]{36}","slug":"')


def _get(path):
    # The whole trick: ask Next.js for the RSC flight payload, not the HTML.
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA, "RSC": "1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"User-Agent": UA, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _scan(raw, pattern, require=()):
    """Pull every JSON object at every match of `pattern` out of the flight payload."""
    dec = json.JSONDecoder()
    out, seen = [], set()
    for m in pattern.finditer(raw):
        try:
            obj, _ = dec.raw_decode(raw, m.start())
        except ValueError:
            continue  # truncated / not a standalone object; skip
        if not isinstance(obj, dict) or not all(k in obj for k in require):
            continue
        key = json.dumps(obj, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(obj)
    return out


# Creator objects (zai, openai, anthropic, ...) share the id+slug shape but have no
# "creator" key. Requiring it is the cheap discriminator. Do NOT hardcode the ratio:
# the page matched ~712 objects with ~59 creators, then ~1312 with ~656 -- the creator
# count is not stable, and a fixed assumption would silently ingest or drop rows.
@functools.lru_cache(maxsize=None)  # 2.9 MB per call; compare() would otherwise refetch it
def catalogue(anchor_slug="glm-5-3-flash"):
    """Full model detail set, taken from any single model page's flight payload."""
    raw = _get(f"/models/{anchor_slug}")
    return tuple(_scan(raw, _MODEL_START, require=("creator",)))


def series(model_ids, prompt_type="medium", host_model_ids=None):
    return _post(
        "/api/models/performance-over-time",
        {
            "endpoint": "/api/models/performance-over-time",
            "modelIds": sorted(model_ids),
            "hostModelIds": sorted(host_model_ids or []),
            "promptType": prompt_type,
        },
    )


def blended(m, cached=0, inp=3, out=1):
    """Blended $/M for ANY cached:input:output mix, derived from component prices.

    AA ships five precomputed ratios; deriving it instead covers every mix and lets
    you solve for the crossover where the ranking flips. Verified in demo() against
    all five published fields. Cache price falls back to the input price when the
    provider publishes none (most models lack cacheHitPrice).
    """
    i, o = m.get("price1mInputTokens"), m.get("price1mOutputTokens")
    if i is None or o is None:
        return None
    ci = m.get("cacheHitPrice")
    if ci is None:
        ci = i
    return (cached * ci + inp * i + out * o) / (cached + inp + out)


def _omniscience(m):
    """Accuracy and hallucination rate, in either shape AA has served them.

    Until 2026-09-22 these were nested in `omniscienceBreakdown`; they are now
    top-level (`omniscienceAccuracy`, `omniscienceHallucinationRate`). Read both --
    a staged rollout can serve either, and a missing key must not silently become
    None, which is what demo() caught when the flattening landed.
    """
    nested = m.get("omniscienceBreakdown") or {}
    return (m.get("omniscienceAccuracy", nested.get("accuracy")),
            m.get("omniscienceHallucinationRate", nested.get("hallucinationRate")))


def _row(m, mix=(0, 3, 1), prompt_type="long"):
    """Turn one catalogue model into a comparison row.

    Pure -- takes a model dict, returns a dict. Pulled out of compare() so the field
    paths can be unit-tested offline. This matters because extraction is the most
    error-prone part of the tool: a wrong path reads None silently, and a metric
    missing from a comparison row looks identical to a model that has no data.
    """
    perf = (m.get("performanceByPromptType") or {}).get(prompt_type) or {}
    var = m.get("outputSpeedVariance") or {}
    accuracy, hallucination = _omniscience(m)
    return {
        "slug": m["slug"],
        "effort": (m.get("effort") or {}).get("label"),
        "intelligenceIndex": m["intelligenceIndex"],
        "iiEstimated": m["intelligenceIndexIsEstimated"],
        "iiSource": m["performanceDataSource"]["type"],
        "deprecated": m["deprecated"],
        # 375 of the 381 deprecated models name their own replacement. Knowing a model
        # is deprecated is a dead end; knowing what superseded it is actionable.
        "supersededBy": m.get("deprecatedTo"),
        "price": blended(m, *mix),
        "speed": perf.get("medianOutputSpeed"),
        "speedP05": var.get("p05"),
        "speedP95": var.get("p95"),
        "hosts": m["hostModelCount"],
        "evals": len(m["intelligenceIndexEvaluations"]),
        "context": m["contextWindowTokens"],
        "params": m.get("parameters"),
        "activeParams": m.get("inferenceParametersActiveBillions"),
        "license": m.get("licenseName"),
        # Total output tokens to run the whole benchmark suite -- this is what the
        # "cost to run the intelligence index" figure is actually spending on.
        "suiteTokens": (m.get("canonicalIntelligenceIndexTokenCount") or {}).get("output"),
        # The rate is CONDITIONAL on answering wrong (share of failures that are
        # confabulations rather than abstentions), so it must be read next to
        # accuracy: 14% hallucination at 9% accuracy is abstention, not honesty.
        # Absolute confabulation = (1 - accuracy) * hallucination.
        "accuracy": accuracy,
        "hallucination": hallucination,
        # briefcaseBreakdown is the only place AA publishes an interval. Without it a
        # 2-point gap between two models looks identical to a 200-point one.
        "elo": ((m.get("briefcaseBreakdown") or {}).get("overall") or {}).get("elo"),
        "eloLo": ((m.get("briefcaseBreakdown") or {}).get("overall") or {}).get("lower95ci"),
        "eloHi": ((m.get("briefcaseBreakdown") or {}).get("overall") or {}).get("upper95ci"),
        "isReasoning": m.get("isReasoning"),
        "reasoningSec": (m.get("endToEndResponseTime") or {}).get("reasoning"),
    }


def compare(slugs, mix=(0, 3, 1), prompt_type="long"):
    """Compare models, flagging the ways the raw numbers mislead.

    `mix` is cached:input:output. Returns (rows, warnings). Every score carries its
    provenance next to it, because in this catalogue most intelligence indices are
    *estimated* rather than measured, most models are deprecated, most have no
    per-eval breakdown at all, and a median speed means little without its p05-p95
    spread and host count. (Counts move daily -- see FINDINGS.md rather than a number
    written here.)
    """
    by_slug = {m["slug"]: m for m in catalogue()}
    rows, warns = [], []
    for s in slugs:
        m = by_slug.get(s)
        if m is None:
            warns.append("%s: not in catalogue; closest: %s"
                         % (s, difflib.get_close_matches(s, by_slug, n=3)))
            continue
        rows.append(_row(m, mix, prompt_type))
    for r in rows:
        if r["iiEstimated"]:
            warns.append("%s: intelligence index is ESTIMATED (%s), not measured"
                         % (r["slug"], r["iiSource"]))
        if r["deprecated"]:
            warns.append("%s: deprecated%s" % (
                r["slug"], " -- superseded by %s" % r["supersededBy"]
                if r["supersededBy"] else ""))
        if not r["evals"]:
            warns.append("%s: no per-eval breakdown behind its index" % r["slug"])
        if r["hosts"] is not None and r["hosts"] < 3:
            warns.append("%s: speed/price rest on %d host(s) -- thin sample"
                         % (r["slug"], r["hosts"]))
        if r["price"] is None:
            warns.append("%s: no published API price -- excluded from any cost ranking"
                         % r["slug"])
        if r["hallucination"] is None:
            warns.append("%s: no omniscience/hallucination data" % r["slug"])
    # A reasoning model reporting 0s of reasoning time has not had its reasoning timed --
    # the cost is silently folded into "input", which is why some TTFC figures are absurd.
    for r in rows:
        if r["isReasoning"] and r["reasoningSec"] == 0:
            warns.append("%s: reasoning time reported as 0s -- latency excludes reasoning, "
                         "so its TTFC and total are not comparable" % r["slug"])

    # Adjacent models whose 95% intervals overlap are not actually separated.
    rated = sorted((r for r in rows if r["elo"] is not None),
                   key=lambda r: -r["elo"])
    for a, b in zip(rated, rated[1:]):
        if a["eloLo"] <= b["eloHi"] and b["eloLo"] <= a["eloHi"]:
            warns.append("%s (%d) and %s (%d) have overlapping 95%% CIs -- "
                         "statistically indistinguishable, do not rank them apart"
                         % (a["slug"], a["elo"], b["slug"], b["elo"]))

    efforts = {r["effort"] for r in rows if r["effort"]}
    if len(efforts) > 1:
        warns.append("mixed effort levels %s -- not apples-to-apples" % sorted(efforts))
    return rows, warns


# The same model ships as several slugs, one per effort level (claude-fable-5-1,
# -xhigh, -high, -medium). Ranking the raw catalogue counts one model four times.
_EFFORT_SUFFIXES = ("-minimal", "-low", "-medium", "-high", "-xhigh", "-max")


def _base_slug(slug):
    for suf in _EFFORT_SUFFIXES:
        if slug.endswith(suf):
            return slug[: -len(suf)]
    return slug


def trustworthy(dedupe_effort=True):
    """The models worth comparing: real eval breakdown, not deprecated, not estimated.

    That is roughly one model in six. Collapses effort variants to the top-scoring one (highest
    intelligence index, tie-broken on host count) so one model appears once.
    """
    pool = [
        m for m in catalogue()
        if m["intelligenceIndexEvaluations"] and not m["deprecated"]
        and not m["intelligenceIndexIsEstimated"]
    ]
    if not dedupe_effort:
        return sorted(pool, key=lambda m: -(m["intelligenceIndex"] or 0))
    best = {}
    for m in pool:
        b = _base_slug(m["slug"])
        rank = lambda x: (x["intelligenceIndex"] or 0, x["hostModelCount"] or 0)
        if b not in best or rank(m) > rank(best[b]):
            best[b] = m
    return sorted(best.values(), key=lambda m: -(m["intelligenceIndex"] or 0))


def composite(slugs, weights=None):
    """Weighted composite over the evals behind the intelligence index.

    Raw scores are incomparable (aa-briefcase ~1460, automationbench ~0.60), so each
    eval is min-max normalised across `slugs` first -- which makes the result relative
    to the compared set: adding a model moves everyone. `weights` maps eval slug to
    weight; None means equal. A model with no score for an eval counts as worst.
    """
    by = {m["slug"]: m for m in catalogue()}
    table = {}
    for s in slugs:
        m = by.get(s)
        if m is None:
            continue
        for e in m["intelligenceIndexEvaluations"]:
            table.setdefault(e["slug"], {})[s] = e["score"]
    scores = {s: 0.0 for s in slugs if s in by}
    wsum = 0.0
    for ev, sc in table.items():
        lo, hi = min(sc.values()), max(sc.values())
        w = 1.0 if weights is None else weights.get(ev, 0.0)
        if not w or hi == lo:
            continue
        for s, v in sc.items():
            scores[s] += w * (v - lo) / (hi - lo)
        wsum += w
    if wsum:
        scores = {s: v / wsum for s, v in scores.items()}
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def _ranks(values, high_better, ties=()):
    """slug -> 1-based rank. Ranking sidesteps incomparable units entirely.

    `ties` is a sequence of slug groups whose confidence intervals overlap. Members share
    the average of their positions, so a tie costs you nothing numerically but the
    ordering stops claiming a separation the data does not support.
    """
    order = sorted(values, key=lambda s: -values[s] if high_better else values[s])
    rank = {s: i + 1 for i, s in enumerate(order)}
    for group in ties:
        members = [s for s in order if s in group]
        if len(members) < 2:
            continue
        shared = sum(rank[s] for s in members) / len(members)
        for s in members:
            rank[s] = shared
    return rank


def _ci_ties(rows):
    """Group models whose 95% intervals overlap.

    Overlap is chained, so a run of mutually-adjacent overlaps becomes one group -- a
    deliberate simplification, and conservative: it only ever merges, never reorders.
    """
    rated = sorted((r for r in rows if r["elo"] is not None), key=lambda r: -r["elo"])
    groups, cur = [], []
    for r in rated:
        if cur and r["eloHi"] >= cur[-1]["eloLo"]:
            cur.append(r)
        else:
            if cur:
                groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    return [frozenset(r["slug"] for r in g) for g in groups if len(g) > 1]


def _axis_score(axes, slug, axis_weights=None):
    """Weighted mean of a model's per-axis ranks -- or None if it is missing any axis.

    Averaging over only the axes a model happens to have would score missing data as
    good data: a quality-only model would outrank one that is top-3 on every axis.
    That was a real bug (the no-price k2-horizon models scored on 1 axis against
    models scored on 3). Returns (score, per-axis ranks); score is None whenever the
    model is not measured on every axis in `axes`. Pulled out of rank() so the rule
    is testable offline.
    """
    items = [(n, m[slug]) for n, m in axes if slug in m]
    ranks = {n: r for n, r in items}
    if not items or len(items) < len(axes):
        return None, ranks
    w = {n: (axis_weights or {}).get(n, 1.0) for n, _ in items}
    return sum(w[n] * r for n, r in items) / sum(w.values()), ranks


def rank(slugs, mix=(0, 3, 1), weights=None, axis_weights=None, with_stability=True):
    """Collapse every axis into one ordering: weighted mean of per-axis RANKS.

    Default is equal weight across quality / cost / speed / stability, which is an
    assumption, not a finding -- pass axis_weights={"cost": 3} to shift it. `weights`
    (eval weights) swaps Artificial Analysis's index for composite() as the quality
    axis. Models missing an axis keep their per-axis ranks but get score=None rather
    than an average over the axes they do have -- averaging over fewer axes would
    reward missing data. `axes` in each row shows what that model was measured on.
    """
    rows, _ = compare(slugs, mix)
    present = [r["slug"] for r in rows]
    qual = (composite(present, weights) if weights
            else {r["slug"]: r["intelligenceIndex"] for r in rows})
    # Only the quality axis has published intervals, so only it can hold a tie.
    axes = [("quality", _ranks(qual, True, _ci_ties(rows))),
            ("cost", _ranks({r["slug"]: r["price"] for r in rows if r["price"] is not None}, False)),
            ("speed", _ranks({r["slug"]: r["speed"] for r in rows if r["speed"]}, True))]
    if with_stability:
        st = {s["slug"]: s["swingPct"] for s in stability(present)}
        if st:
            axes.append(("stability", _ranks(st, False)))
    out = []
    for s in present:
        score, ax = _axis_score(axes, s, axis_weights)
        if not ax:
            continue
        out.append({"slug": s, "score": score, "axes": ax})
    out.sort(key=lambda r: (r["score"] is None, r["score"] or 0))
    return out


def stability(slugs, prompt_type="long"):
    """7-day median-speed history per model, so a point-in-time figure can be checked.

    Anonymous access is capped: series() reports restrictedByPlan=True, planLimitDays=7.
    So this measures *stability*, not longevity — enough to catch a model whose
    headline speed is moving, not enough to establish a trend. Sorted most volatile first.
    """
    by = {m["slug"]: m for m in catalogue()}
    ids = [by[s]["id"] for s in slugs if s in by]
    slug_of = {by[s]["id"]: s for s in slugs if s in by}
    rows = []
    for ms in series(ids, prompt_type)["modelSeries"]:
        speeds = [p["medianOutputSpeed"] for p in ms["points"] if p.get("medianOutputSpeed")]
        if not speeds:
            continue
        med = statistics.median(speeds)
        rows.append({
            "slug": slug_of.get(ms["modelId"], ms["modelId"]),
            "days": len(speeds),
            "median": med,
            "lo": min(speeds),
            "hi": max(speeds),
            "swingPct": 100 * (max(speeds) - min(speeds)) / med,
            "driftPct": 100 * (speeds[-1] - speeds[0]) / speeds[0],
        })
    rows.sort(key=lambda r: -r["swingPct"])
    return rows


def eval_grid(slugs):
    """Rank the models inside each eval that composes the intelligence index.

    The composite hides what you actually care about, and the 10 evals are on
    incomparable scales (aa-briefcase ~1460, automationbench ~0.60), so rank within
    the set rather than comparing raw scores across evals.
    """
    by = {m["slug"]: m for m in catalogue()}
    table = {}
    for s in slugs:
        m = by.get(s)
        if m is None:
            continue
        for e in m["intelligenceIndexEvaluations"]:
            table.setdefault(e["slug"], {})[s] = e["score"]
    if not table:
        return "no per-eval data for any of: %s" % slugs
    wins = {s: 0 for s in slugs}
    hdr = "%-34s" % "eval" + "".join("%8s" % s[:7] for s in slugs)
    lines = [hdr, "-" * len(hdr)]
    for ev in sorted(table):
        order = sorted(table[ev], key=lambda s: -table[ev][s])
        wins[order[0]] += 1
        lines.append("%-34s" % ev[:34] + "".join(
            "%8s" % ("#%d" % (order.index(s) + 1) if s in table[ev] else "-")
            for s in slugs))
    lines.append("")
    lines.append("wins: " + ", ".join("%s=%d" % (s, wins[s]) for s in slugs))
    return "\n".join(lines)


# Benchmark fields AA publishes alongside the composite index. Most are sparsely
# populated -- many are null for most models -- so score_grid() shows only the ones
# with data for at least one of the compared models, and names the rest.
BENCHMARK_FIELDS = (
    "gpqa", "hle", "aime25", "livecodebench", "scicode", "terminalBench21",
    "terminalbenchHard", "terminalBench40", "tau2", "tauBanking", "mmmuPro",
    "gdpval", "gdpvalNormalized", "critpt", "apexAgents", "ifbench", "lcr",
    "mlcrOverall", "harveyLab", "itBenchSre", "analystAgent", "omniscience",
    "automationBenchPartialScore", "gdpPdfAllPass", "enterpriseOpsGym",
)


def score_grid(slugs):
    """All benchmark scores side by side: the 10 composite evals plus every standalone
    benchmark field with data.

    eval_grid() ranks within each eval, which is right for ordering but hides the size
    of the gaps: 1526 vs 1461 and 1526 vs 1525 both render as "#1 vs #2". Scores are on
    mixed scales (ELO ~1500, percentages 0-1, omniscience an index around -10..20), so
    they are formatted per range rather than normalised.

    Several standalone fields duplicate composite ones under different names (for
    example `hle` and `humanitys-last-exam`, `gdpval` and `gdpval-aa`). They are kept
    because agreement between the two is a free sanity check on the extraction.
    """
    by = {m["slug"]: m for m in catalogue()}
    cols = [s for s in slugs if s in by]
    if not cols:
        return "no such models: %s" % slugs
    hdr = "%-42s" % "eval" + "".join("%15s" % s[:14] for s in slugs)
    sep = "-" * len(hdr)
    lines = []

    def row(label, get):
        cells = []
        for s in slugs:
            if s not in by:
                cells.append("%15s" % "-")
                continue
            v = get(by[s])
            if v is None:
                cells.append("%15s" % "-")
            elif 0 <= v <= 1:
                cells.append("%14.1f%%" % (100 * v))
            elif abs(v) >= 1000:
                cells.append("%15s" % format(round(v), ","))
            else:
                cells.append("%15.2f" % v)
        lines.append("%-42s" % label[:42] + "".join(cells))

    composite = {}
    for s in cols:
        for e in by[s]["intelligenceIndexEvaluations"]:
            composite.setdefault(e["slug"], {})[s] = e["score"]
    lines.append("composite: the 10 evals behind the intelligence index")
    lines.append(hdr)
    lines.append(sep)
    for ev in sorted(composite):
        row(ev, lambda m, ev=ev: composite[ev].get(m["slug"]))

    # aa-briefcase is both the most-quoted eval and the only one with a published
    # interval. A gap of 1526 vs 1461 and one of 1526 vs 1525 look identical as scores,
    # so say whether the difference survives the error bars.
    iv = []
    for s in cols:
        bb = ((by[s].get("briefcaseBreakdown") or {}).get("overall") or {})
        if None not in (bb.get("elo"), bb.get("lower95ci"), bb.get("upper95ci")):
            iv.append({"slug": s, "elo": bb["elo"],
                       "eloLo": bb["lower95ci"], "eloHi": bb["upper95ci"]})
    if len(iv) >= 2:
        lines.append("")
        lines.append("aa-briefcase 95% interval: " + ", ".join(
            "%s %d [%d, %d]" % (r["slug"], r["elo"], r["eloLo"], r["eloHi"])
            for r in sorted(iv, key=lambda r: -r["elo"])))
        lines.append("  -> " + ("OVERLAP: treat as a tie, do not rank them apart"
                                if _ci_ties(iv) else
                                "separated: this gap is real"))

    extra = {}
    for k in BENCHMARK_FIELDS:
        vals = {s: by[s].get(k) for s in cols}
        if any(v is not None for v in vals.values()):
            extra[k] = vals
    if extra:
        lines.append("")
        lines.append("standalone benchmarks (only those with data for one of these)")
        lines.append(hdr)
        lines.append(sep)
        for k in BENCHMARK_FIELDS:
            if k in extra:
                row(k, lambda m, k=k: m.get(k))
    missing = [k for k in BENCHMARK_FIELDS if k not in extra]
    if missing:
        lines.append("")
        lines.append("no data for either: " + ", ".join(missing))
    return "\n".join(lines)


def demo():
    """Self-check: parsing works and both kept channels answer."""
    models = catalogue()
    assert len(models) > 500, f"expected full catalogue, got {len(models)}"
    assert len({m["slug"] for m in models}) == len(models), "duplicate slugs"
    for m in models[:5]:
        assert m["slug"] and m["name"] and "intelligenceIndex" in m
    by_slug = {m["slug"]: m for m in models}
    for s in ("glm-5-3-flash", "deepseek-v4-1-flash", "gpt-5-6-luna", "gemini-3-8-flash", "glm-5-3"):
        assert s in by_slug, f"{s} missing from catalogue"

    # Still guards the headline claim in the writeup: the comparisons page ignores
    # ?compare= server-side. Cheap to re-verify, and it is the easiest thing to break.
    assert _get("/models/comparisons?compare=deepseek-v4-1-flash") == _get(
        "/models/comparisons?compare=gpt-5-5,claude-opus-4-6"
    ), "compare param now changes server output -- re-check client-side filtering"

    ids = [by_slug["glm-5-3-flash"]["id"], by_slug["gpt-5-6-luna"]["id"]]
    s = series(ids, "medium")
    assert len(s["modelSeries"]) == len(ids)
    assert all(p["intervalStartDate"] for ms in s["modelSeries"] for p in ms["points"])

    # The guards are the whole point of compare(), so assert they actually fire.
    rows, warns = compare(["glm-5-3", "gemini-3-8-flash", "glm-4-5v"])
    assert len(rows) == 3
    assert any("mixed effort" in w for w in warns), "effort guard did not fire"
    assert any("ESTIMATED" in w for w in warns), "estimated-index guard did not fire"
    # Hallucination is nested and easy to lose; assert it survives into the row.
    hal = {r["slug"]: r["hallucination"] for r in rows}
    assert hal["glm-5-3"] is not None, "hallucination rate missing from compare()"
    assert 0 < hal["glm-5-3"] < 1, hal
    # Luna reports 0s reasoning on a reasoning model: the timing-gap guard must fire.
    assert any("reasoning time reported as 0s" in w for w in warns), warns
    assert any(r["elo"] for r in rows), "no confidence intervals parsed"
    assert "aa-briefcase" in score_grid(["glm-5-3"]), "score grid lost the eval slugs"
    # blended() replaces the five published ratios, so it must reproduce all of them
    # exactly. If AA changes its price model, this is where we find out.
    checked = 0
    for m in models:
        for field, mix in (("price1mBlended0To3To1", (0, 3, 1)),
                           ("price1mBlended7To2To1", (7, 2, 1)),
                           ("price1mBlended0To1To1", (0, 1, 1)),
                           ("price1mBlended100To1To1", (100, 1, 1)),
                           ("price1mBlended0To100To1", (0, 100, 1))):
            if m[field] is not None:
                assert abs(blended(m, *mix) - m[field]) < 1e-9, \
                    f"{m['slug']}: blended{mix}={blended(m, *mix)} != {field}={m[field]}"
                checked += 1
    assert checked > 1000, f"only validated {checked} published prices"

    # Price ranking must depend on the mix, or --mix is pointless.
    cheap = lambda mix: min(compare(["deepseek-v4-1-flash", "glm-5-3-flash"], mix)[0],
                            key=lambda r: r["price"])["slug"]
    assert cheap((0, 3, 1)) != cheap((100, 1, 1)), "mix no longer changes the ranking"

    # Rankable subset: must shrink on dedupe and hold no two slugs from one family.
    full, deduped = trustworthy(False), trustworthy()
    assert 50 < len(deduped) < len(full), f"trustworthy subset looks wrong: {len(deduped)}/{len(full)}"
    assert len({_base_slug(m["slug"]) for m in deduped}) == len(deduped), "effort dedupe missed a family"

    # The CLI parser is the only path demo() does NOT naturally exercise, so it was
    # broken silently twice before. Pulled it out into _parse_args (which also
    # resolves slugs, since the dropped `slugs =` line was the actual break) and
    # hit it here without the network -- a dropped variable now fails demo, not users.
    parsed = _parse_args(["glm-5-3", "--mix", "0:3:1"])
    assert parsed[0] == ["glm-5-3"] and parsed[1] == (0, 3, 1) and parsed[2] is None, parsed
    parsed = _parse_args(["--weights", "scicode=2,terminalbench-4-0=3"])
    assert parsed[2] == {"scicode": 2.0, "terminalbench-4-0": 3.0}, parsed
    for bad in (["--mix", "bogus"], ["--mix", "0:3"], ["--weights", "scicode"],
                ["--weights", "=2"], ["--weights", "scicode=2,bad=x"]):
        try:
            _parse_args(bad)
        except SystemExit:
            continue
        raise AssertionError("bad args should have exited: %r" % bad)

    # Weighting a single eval must promote that eval's own winner to the top. Deriving
    # the expected winner from the grid keeps this about the logic, not today's data.
    trio = ["glm-5-3", "gemini-3-8-flash", "deepseek-v4-1-flash"]
    c = composite(trio)
    assert len(c) == 3 and all(0 <= v <= 1 for v in c.values()), c
    for ev in ("humanitys-last-exam", "terminalbench-4-0", "omniscience"):
        by_ev = {m["slug"]: {e["slug"]: e["score"]
                             for e in m["intelligenceIndexEvaluations"]}
                 for m in catalogue() if m["slug"] in trio}
        top = max((s for s in trio if ev in by_ev[s]), key=lambda s: by_ev[s][ev])
        assert list(composite(trio, {ev: 1}))[0] == top, f"{ev}: weighting did not promote its winner"

    # rank() must actually reorder: equal weights vs cost-heavy must disagree, or the
    # collapsed answer is not carrying the cost axis at all.
    five = ["glm-5-3", "glm-5-3-flash", "gemini-3-8-flash", "deepseek-v4-1-flash", "gpt-5-6-luna"]
    flat = rank(five)
    cheap = rank(five, axis_weights={"cost": 4})
    assert [r["slug"] for r in flat] != [r["slug"] for r in cheap], "cost axis had no effect"
    assert len(flat) == len(five) and all(r["axes"] for r in flat)

    # _ranks is pure, so ties can be tested without waiting on overlapping real models.
    plain = _ranks({"a": 3, "b": 2, "c": 1}, True)
    assert plain == {"a": 1, "b": 2, "c": 3}, plain
    tied = _ranks({"a": 3, "b": 2, "c": 1}, True, ties=[frozenset({"a", "b"})])
    assert tied["a"] == tied["b"] == 1.5 and tied["c"] == 3, tied

    st = stability(["deepseek-v4-1-flash", "gpt-5-6-luna"])
    assert len(st) == 2 and all(0 < r["days"] <= 7 for r in st), f"stability looks wrong: {st}"
    assert all(r["swingPct"] >= 0 for r in st)

    print("ok: catalogue=%d models, series=%d series, compare=%d rows/%d warnings"
          % (len(models), len(s["modelSeries"]), len(rows), len(warns)))


def _parse_args(rest, default_slugs=None):
    """Arg parser for compare/trustworthy/rank, pulled out so demo() can exercise it.

    The previous inline version was easy to break silently: a dropped variable once
    shipped a NameError to every CLI user because demo() never exercised this path.
    Now it is reachable from demo(), every parser change is checked.
    """
    rest = list(rest)
    as_json = "--json" in rest
    show_evals = "--evals" in rest
    show_stab = "--stability" in rest
    rest = [a for a in rest
            if a not in ("--json", "--evals", "--stability", "--rank", "--scores")]
    if "--mix" in rest:
        i = rest.index("--mix")
        parts = rest[i + 1].split(":")
        if len(parts) != 3 or not all(p.lstrip("+").isdigit() for p in parts):
            sys.exit("--mix must be three non-negative integers like 0:3:1")
        mix, rest = tuple(int(p) for p in parts), rest[:i] + rest[i + 2:]
    else:
        mix = (0, 3, 1)
    weights = None
    if "--weights" in rest:
        i = rest.index("--weights")
        weights = {}
        for kv in rest[i + 1].split(","):
            k, _, v = kv.partition("=")
            if not k or not v:
                sys.exit("--weights expects eval-slug=number pairs, e.g. "
                         "--weights scicode=2,terminalbench-4-0=3")
            try:
                weights[k.strip()] = float(v)
            except ValueError:
                sys.exit("--weights expects eval-slug=number pairs, e.g. "
                         "--weights scicode=2,terminalbench-4-0=3")
        rest = rest[:i] + rest[i + 2:]
    # Resolving slugs here (not in __main__) matters: the dropped `slugs =` line was
    # what broke the CLI, and this is the only place demo() can reach it. default_slugs
    # lets tests inject slugs without hitting the network.
    slugs = rest or list(default_slugs if default_slugs is not None
                         else (m["slug"] for m in trustworthy()))
    return slugs, mix, weights, as_json, show_evals, show_stab


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "demo":
        demo()
    elif args[0] == "catalogue":
        json.dump(catalogue(), open(args[1], "w"), indent=1)
        print("wrote", args[1])
    elif args[0] in ("compare", "trustworthy", "rank"):
        slugs, mix, weights, as_json, show_evals, show_stab = _parse_args(args[1:])
        show_rank = args[0] == "rank" or "--rank" in args[1:]
        show_scores = "--scores" in args[1:]
        rows, warns = compare(slugs, mix)
        rows.sort(key=lambda r: -(r["intelligenceIndex"] or 0))
        if as_json:
            print(json.dumps({"mix": "%d:%d:%d" % mix, "models": rows,
                              "warnings": warns}, indent=1))
        else:
            hdr = ("%-22s %-6s %6s %8s %7s %-13s %5s %7s %6s %8s" %
                   ("slug", "effort", "II", "$/M", "tok/s", "p05-p95", "hosts",
                    "halluc%", "acc", "context"))
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                spread = ("%3.0f-%-4.0f" % (r["speedP05"], r["speedP95"])
                          if r["speedP05"] else "-")
                ii = "%6.2f%s" % (r["intelligenceIndex"], "~" if r["iiEstimated"] else " ")
                price = "     n/a" if r["price"] is None else "%8.3f" % r["price"]
                hal = ("%6.1f%%" % (100 * r["hallucination"])
                       if r["hallucination"] is not None else "     -")
                acc = "%6.3f" % r["accuracy"] if r["accuracy"] is not None else "     -"
                # Rendered in thousands to match the site's own display, so the
                # Kimi-vs-others gap (1049k vs 1000k) is actually visible.
                ctx = ("%7dk" % round(r["context"] / 1000)
                       if r["context"] else "      -")
                print("%-22s %-6s %s %s %7.1f %-13s %5s %7s %6s %8s" %
                      (r["slug"], r["effort"] or "-", ii, price, r["speed"] or 0,
                       spread, r["hosts"], hal, acc, ctx))
            print("\n$/M at %d:%d:%d (cached:in:out). ~ = estimated index." % mix)
            print("halluc% is the share of WRONG answers that are confabulations, not a "
                  "share of all answers -- read it next to acc.")
            for w in warns:
                print("WARN:", w)
            if show_rank:
                print("one ordering = weighted mean of per-axis ranks (lower is better)")
                print("%-24s %6s  %s" % ("slug", "score", "per-axis ranks"))
                for r in rank(slugs, mix, weights):
                    ax = "  ".join("%s=#%d" % (k, v) for k, v in r["axes"].items())
                    print("%-24s %6s  %s" % (r["slug"],
                          "-" if r["score"] is None else "%.2f" % r["score"], ax))
            if show_evals:
                print("\n" + eval_grid(slugs))
            if show_scores:
                print("\n" + score_grid(slugs))
            if show_evals or weights:
                print("\nweighted composite (normalised within this set)"
                      + ("" if weights else ", equal weights"))
                for s, v in composite(slugs, weights).items():
                    print("  %-22s %.3f" % (s, v))
            if show_stab:
                print("\n7-day speed stability (planLimitDays=7 caps anonymous history)")
                for r in stability(slugs):
                    flag = ""
                    if r["swingPct"] > 40:
                        flag = "  <-- volatile"
                    if abs(r["driftPct"]) > 30:
                        flag += "  <-- trending %s" % ("up" if r["driftPct"] > 0 else "down")
                    print("  %-22s %2dd  med %6.1f  %6.1f-%-6.1f  swing %5.1f%%  drift %+6.1f%%%s"
                          % (r["slug"], r["days"], r["median"], r["lo"], r["hi"],
                             r["swingPct"], r["driftPct"], flag))
    else:
        sys.exit(__doc__)
