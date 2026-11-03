"""Harder benchmark: many decisions per call, long transcripts, and concurrent load.

Usage:
    uv run python bench/hard.py accuracy --runs 3 [--verbose]
    uv run python bench/hard.py batch --runs 3
    uv run python bench/hard.py load --concurrency 1,8,24 --rounds 3

Every mode accepts --decision-models and --llms, as in bench/run.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env.local")

from livekit.agents import llm, utils  # noqa: E402

from signals import (  # noqa: E402
    CONFIRMED_PROMPT,
    FRUSTRATION_LEVELS,
    FRUSTRATION_PROMPT,
    HANDOFF_PROMPT,
    INTENT_OPTIONS,
    INTENT_PROMPT,
    DecisionClassifier,
    LLMClassifier,
)

# 16 decisions answered in one call, like a production guardrail and routing batch.
DECISIONS: dict[str, llm.Decision] = {
    "handoff": llm.Probability(HANDOFF_PROMPT),
    "intent": llm.Choice(INTENT_PROMPT, INTENT_OPTIONS),
    "frustration": llm.Score(FRUSTRATION_PROMPT, FRUSTRATION_LEVELS),
    "confirmed": llm.Probability(CONFIRMED_PROMPT),
    "next_action": llm.Choice(
        "What should the agent do next, based on the caller's latest turn?",
        {
            "continue": "Ask the next application question or answer the caller.",
            "read_back": "Read back a phone number or ZIP code the caller just gave.",
            "transfer": "Connect the caller with a person.",
            "safety_exit": "End the call politely because the caller made a threat.",
            "route_claims": "Send the caller to the claims team about a current policy.",
        },
    ),
    "abuse": llm.Probability(
        "In their latest turn, the caller insults, harasses, or swears at the agent."
    ),
    "threat": llm.Probability(
        "In their latest turn, the caller threatens violence or damage to a person or property."
    ),
    "prompt_injection": llm.Probability(
        "In their latest turn, the caller tries to change the agent's instructions or role, "
        "or asks the agent to reveal its instructions."
    ),
    "existing_claim": llm.Probability(
        "In their latest turn, the caller wants to report or discuss a claim on an "
        "insurance policy they have now."
    ),
    "sensitive_number": llm.Probability(
        "In their latest turn, the caller says a full Social Security number, bank "
        "account number, or credit card number."
    ),
    "competitor_quote": llm.Probability(
        "In their latest turn, the caller mentions a price or quote from another "
        "insurance company."
    ),
    "other_language": llm.Probability(
        "In their latest turn, the caller asks to continue in a language other than English."
    ),
    "callback_time": llm.Probability(
        "In their latest turn, the caller gives a specific day or time for a call back."
    ),
    "stop_application": llm.Probability(
        "In their latest turn, the caller wants to stop or cancel the application."
    ),
    "high_risk_activity": llm.Probability(
        "In their latest turn, the caller describes business activity that involves "
        "alcohol, firearms, roofing, or demolition."
    ),
    "repeated_question": llm.Probability(
        "In their latest turn, the caller says they already answered the agent's question."
    ),
}
PROBABILITIES = [n for n, d in DECISIONS.items() if isinstance(d, llm.Probability)]
THRESHOLD = 0.5


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    cases = []
    for case in data["cases"]:
        prefix = data["prefixes"][case["prefix"]]
        lines = [prefix] if prefix else []
        lines += [f"assistant: {case['agent']}", f"user: {case['user']}"]
        cases.append({**case, "transcript": "\n".join(lines)})
    return cases


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def make_classifiers(args) -> list:
    return [DecisionClassifier(m) for m in args.decision_models.split(",") if m] + [
        LLMClassifier(m) for m in args.llms.split(",") if m
    ]


async def warm_up(classifier, transcript: str) -> str | None:
    try:
        await classifier.evaluate(transcript, DECISIONS)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


# --- accuracy -------------------------------------------------------------------


def score_case(case: dict, values: dict) -> dict:
    """Compare one answer to the labels. Returns per-decision results."""
    out: dict[str, dict] = {}
    # A declined decision counts as a miss: the agent gets no answer for it.
    for name in PROBABILITIES:
        if name in case.get("skip", []):
            continue
        label = name in case["true"]
        if name not in values:
            out[name] = {"ok": False, "brier": 1.0, "p": None, "declined": True}
            continue
        p = float(values[name])
        out[name] = {"ok": (p >= THRESHOLD) == label, "brier": (p - label) ** 2, "p": p}
    for name in ("intent", "next_action"):
        if case.get(name) is not None:
            got = values.get(name)
            out[name] = {"ok": got == case[name], "got": got, "declined": got is None}
    if case.get("frustration") is not None:
        if "frustration" not in values:
            out["frustration"] = {
                "ok": False,
                "err": 2.0,
                "got": None,
                "declined": True,
            }
            return out
        err = abs(float(values["frustration"]) - case["frustration"])
        out["frustration"] = {
            "ok": err <= 0.5,
            "err": err,
            "got": values["frustration"],
        }
    return out


async def run_accuracy(args) -> None:
    cases = load_cases(Path(args.cases))
    report = []
    async with utils.http_context.open():
        for classifier in make_classifiers(args):
            print(f"\n{classifier.name}")
            if error := await warm_up(classifier, cases[0]["transcript"]):
                print(f"  ERROR {error}")
                report.append({"name": classifier.name, "error": error})
                continue
            rows = []
            for run in range(args.runs):
                for case in cases:
                    try:
                        e = await classifier.evaluate(case["transcript"], DECISIONS)
                    except Exception as ex:
                        rows.append({"id": case["id"], "run": run, "error": str(ex)})
                        if args.verbose:
                            print(f"  {case['id']:<26} ERROR {ex}")
                        continue
                    scored = score_case(case, e.values)
                    rows.append(
                        {
                            "id": case["id"],
                            "run": run,
                            "latency": e.latency,
                            "out_tokens": e.output_tokens,
                            "declined": list(e.errors),
                            "scored": scored,
                        }
                    )
                    if args.verbose:
                        misses = [
                            f"{n} declined"
                            if v.get("declined")
                            else f"{n} {v.get('p', v.get('got'))!s:.6}"
                            for n, v in scored.items()
                            if not v["ok"]
                        ]
                        status = "ok  " if not misses else "MISS"
                        print(
                            f"  {case['id']:<26} {status} {e.latency * 1000:6.0f} ms  {'; '.join(misses)}"
                        )
            report.append(summarize_accuracy(classifier.name, rows, cases))

    print_accuracy(report)
    save("hard-accuracy", report)


def summarize_accuracy(name: str, rows: list[dict], cases: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"name": name, "error": rows[0]["error"]}

    def acc(decisions: list[str], subset=None) -> float:
        hits = [
            r["scored"][d]["ok"]
            for r in ok
            if subset is None or r["id"] in subset
            for d in decisions
            if d in r["scored"]
        ]
        return sum(hits) / len(hits) if hits else float("nan")

    guard = ["abuse", "threat", "prompt_injection", "sensitive_number"]
    # Recall: of the labeled-true probability decisions, how many were caught.
    labels = {c["id"]: set(c["true"]) for c in cases}
    positives = [
        r["scored"][d]["ok"]
        for r in ok
        for d in PROBABILITIES
        if d in r["scored"] and d in labels[r["id"]]
    ]
    negatives = [
        not r["scored"][d]["ok"]
        for r in ok
        for d in PROBABILITIES
        if d in r["scored"] and d not in labels[r["id"]]
    ]
    # Flip rate: share of (case, decision) pairs whose thresholded answer changed
    # between runs.
    by_case: dict[str, list[dict]] = {}
    for r in ok:
        by_case.setdefault(r["id"], []).append(r["scored"])
    flips = total = 0
    for runs in by_case.values():
        if len(runs) < 2:
            continue
        for d in runs[0]:
            answers = {
                (v[d]["p"] >= THRESHOLD)
                if v[d].get("p") is not None
                else v[d].get("got")
                for v in runs
            }
            total += 1
            flips += len(answers) > 1
    long_ids = {c["id"] for c in cases if c["prefix"] == "long"}
    short_ids = {c["id"] for c in cases if c["prefix"] == "none"}
    paired = {i.removesuffix("-short") for i in short_ids}
    latencies = [r["latency"] for r in ok]
    return {
        "name": name,
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "declined": sum(len(r["declined"]) for r in ok),
        "prob_acc": acc(PROBABILITIES),
        "recall": sum(positives) / len(positives),
        "false_pos": sum(negatives) / len(negatives),
        "brier": statistics.mean(
            r["scored"][d]["brier"]
            for r in ok
            for d in PROBABILITIES
            if d in r["scored"]
        ),
        "guardrail_acc": acc(guard),
        "intent_acc": acc(["intent"]),
        "next_action_acc": acc(["next_action"]),
        "frustration_acc": acc(["frustration"]),
        "long_acc": acc(list(DECISIONS), paired & long_ids),
        "short_acc": acc(list(DECISIONS), short_ids),
        "flip_rate": flips / total if total else float("nan"),
        "p50_ms": pct(latencies, 0.5) * 1000,
        "p95_ms": pct(latencies, 0.95) * 1000,
        "out_tokens": statistics.mean(r["out_tokens"] or 0 for r in ok),
        "rows": rows,
    }


def print_accuracy(report: list[dict]) -> None:
    cols = [
        ("prob_acc", "prob acc", "%"),
        ("recall", "recall", "%"),
        ("false_pos", "false +", "%"),
        ("brier", "Brier", ".3f"),
        ("guardrail_acc", "guard", "%"),
        ("intent_acc", "intent", "%"),
        ("next_action_acc", "next act", "%"),
        ("frustration_acc", "frust", "%"),
        ("long_acc", "long", "%"),
        ("short_acc", "short", "%"),
        ("flip_rate", "flips", "%"),
        ("p50_ms", "p50 ms", ".0f"),
        ("p95_ms", "p95 ms", ".0f"),
        ("out_tokens", "out tok", ".0f"),
    ]
    header = f"{'classifier':<30} {'n':>4} {'err':>4} {'decl':>4} " + " ".join(
        f"{label:>8}" for _, label, _ in cols
    )
    print("\n" + header + "\n" + "-" * len(header))
    for r in report:
        if "n" not in r:
            print(f"{r['name']:<30} ERROR {r['error']}")
            continue
        cells = []
        for key, _, fmt in cols:
            v = r[key]
            cells.append(f"{v:>8.0%}" if fmt == "%" else f"{v:>8{fmt}}")
        print(
            f"{r['name']:<30} {r['n']:>4} {r['errors']:>4} {r['declined']:>4} "
            + " ".join(cells)
        )
    print(
        f"""
prob acc: thresholded at {THRESHOLD} over all {len(PROBABILITIES)} probability decisions. recall: share of true labels caught.
false +: share of false labels flagged. Brier: mean squared error of probabilities (lower is better).
guard: abuse, threat, prompt_injection, and sensitive_number. frust: within 0.5 of the label.
long / short: all decisions on the cases that have both versions, with and without the long conversation.
flips: share of answers that changed between runs of the same case.
err: calls that failed or returned invalid JSON. decl: single decisions the model declined (each counts as a miss)."""
    )


# --- batch size -----------------------------------------------------------------


async def run_batch(args) -> None:
    cases = load_cases(Path(args.cases))
    transcripts = [c["transcript"] for c in cases if c["prefix"] == "long"][:6]
    sizes = [int(k) for k in args.sizes.split(",")]
    names = list(DECISIONS)
    report = []
    async with utils.http_context.open():
        for classifier in make_classifiers(args):
            print(f"\n{classifier.name}")
            if error := await warm_up(classifier, transcripts[0]):
                print(f"  ERROR {error}")
                continue
            for k in sizes:
                subset = {n: DECISIONS[n] for n in names[:k]}
                latencies, tokens, errors = [], [], 0
                for _ in range(args.runs):
                    for t in transcripts:
                        try:
                            e = await classifier.evaluate(t, subset)
                        except Exception:
                            errors += 1
                            continue
                        latencies.append(e.latency)
                        tokens.append(e.output_tokens or 0)
                row = {
                    "name": classifier.name,
                    "decisions": k,
                    "n": len(latencies),
                    "errors": errors,
                    "p50_ms": pct(latencies, 0.5) * 1000 if latencies else None,
                    "p95_ms": pct(latencies, 0.95) * 1000 if latencies else None,
                    "out_tokens": statistics.mean(tokens) if tokens else None,
                }
                report.append(row)
                print(
                    f"  {k:>3} decisions: p50 {row['p50_ms'] or 0:6.0f} ms  "
                    f"p95 {row['p95_ms'] or 0:6.0f} ms  out tok {row['out_tokens'] or 0:5.0f}  "
                    f"errors {errors}/{errors + len(latencies)}"
                )
    save("hard-batch", report)


# --- concurrent load ------------------------------------------------------------


async def run_load(args) -> None:
    cases = load_cases(Path(args.cases))
    transcripts = [c["transcript"] for c in cases if c["prefix"] == "long"]
    levels = [int(c) for c in args.concurrency.split(",")]
    report = []
    async with utils.http_context.open():
        for classifier in make_classifiers(args):
            print(f"\n{classifier.name}")
            if error := await warm_up(classifier, transcripts[0]):
                print(f"  ERROR {error}")
                continue
            for c in levels:
                latencies, errors = [], 0
                for round_ in range(args.rounds):
                    batch = [
                        transcripts[(round_ * c + i) % len(transcripts)]
                        for i in range(c)
                    ]
                    results = await asyncio.gather(
                        *(classifier.evaluate(t, DECISIONS) for t in batch),
                        return_exceptions=True,
                    )
                    for r in results:
                        if isinstance(r, BaseException):
                            errors += 1
                        else:
                            latencies.append(r.latency)
                row = {
                    "name": classifier.name,
                    "concurrency": c,
                    "n": len(latencies),
                    "errors": errors,
                    "p50_ms": pct(latencies, 0.5) * 1000 if latencies else None,
                    "p95_ms": pct(latencies, 0.95) * 1000 if latencies else None,
                    "max_ms": max(latencies) * 1000 if latencies else None,
                }
                report.append(row)
                print(
                    f"  {c:>3} at once: p50 {row['p50_ms'] or 0:6.0f} ms  "
                    f"p95 {row['p95_ms'] or 0:6.0f} ms  max {row['max_ms'] or 0:6.0f} ms  "
                    f"errors {errors}/{errors + len(latencies)}"
                )
    save("hard-load", report)


def save(kind: str, report: list[dict]) -> None:
    out = ROOT / "bench" / "results" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFull results: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["accuracy", "batch", "load"])
    parser.add_argument("--decision-models", default="openai/gpt-6-luna-decisions")
    parser.add_argument("--llms", default="google/gemma-4-31b-it")
    parser.add_argument("--cases", default=str(ROOT / "bench" / "cases_hard.json"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sizes", default="1,4,8,16", help="batch: decisions per call")
    parser.add_argument("--concurrency", default="1,8,24", help="load: calls at once")
    parser.add_argument("--rounds", type=int, default=3, help="load: rounds per level")
    args = parser.parse_args()
    runner = {"accuracy": run_accuracy, "batch": run_batch, "load": run_load}[args.mode]
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
