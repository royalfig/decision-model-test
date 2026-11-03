"""Benchmark a decision model against an LLM on the same per-turn classification.

Usage:
    uv run python bench/run.py
    uv run python bench/run.py --runs 3 --verbose
    uv run python bench/run.py --decision-models openai/gpt-6-luna-decisions,typesafe/jev \\
        --llms google/gemma-4-31b-it,openai/gpt-4.1-mini

Each case in bench/cases.json is a transcript ending on a caller turn, with labels:
handoff (bool), intent (option key, or null to skip), frustration (0 to 2), and
confirmed (bool, or null to skip).
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

from livekit.agents import utils  # noqa: E402

from signals import DecisionClassifier, LLMClassifier, TurnSignals  # noqa: E402

HANDOFF_THRESHOLD = 0.7
CONFIRMED_THRESHOLD = 0.8


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def run_classifier(
    classifier, cases: list[dict], runs: int, verbose: bool
) -> dict:
    # Warm up the connection so the first case doesn't pay for setup.
    try:
        await classifier.classify(cases[0]["transcript"])
    except Exception as e:
        return {"name": classifier.name, "error": f"{type(e).__name__}: {e}"}

    rows = []
    for _ in range(runs):
        for case in cases:
            try:
                s: TurnSignals = await classifier.classify(case["transcript"])
            except Exception as e:
                rows.append({"id": case["id"], "error": f"{type(e).__name__}: {e}"})
                if verbose:
                    print(f"  {case['id']:<26} ERROR {e}")
                continue
            row = {
                "id": case["id"],
                "latency": s.latency,
                "handoff": s.handoff,
                "intent": s.intent,
                "frustration": s.frustration,
                "confirmed": s.confirmed,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "handoff_ok": (s.handoff >= HANDOFF_THRESHOLD) == case["handoff"],
                "intent_ok": None
                if case["intent"] is None
                else s.intent == case["intent"],
                "frustration_err": abs(s.frustration - case["frustration"]),
                "confirmed_ok": None
                if case.get("confirmed") is None
                else (s.confirmed >= CONFIRMED_THRESHOLD) == case["confirmed"],
            }
            rows.append(row)
            if verbose:
                miss = []
                if not row["handoff_ok"]:
                    miss.append(f"handoff {s.handoff:.2f} (want {case['handoff']})")
                if row["intent_ok"] is False:
                    miss.append(f"intent {s.intent} (want {case['intent']})")
                if row["frustration_err"] > 0.75:
                    miss.append(
                        f"frustration {s.frustration:.2f} (want {case['frustration']})"
                    )
                if row["confirmed_ok"] is False:
                    miss.append(
                        f"confirmed {s.confirmed:.2f} (want {case['confirmed']})"
                    )
                status = "ok " if not miss else "MISS"
                print(
                    f"  {case['id']:<26} {status} {s.latency * 1000:6.0f} ms  {'; '.join(miss)}"
                )

    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"name": classifier.name, "error": rows[0]["error"], "rows": rows}
    latencies = [r["latency"] for r in ok]
    intent_rows = [r for r in ok if r["intent_ok"] is not None]
    confirmed_rows = [r for r in ok if r["confirmed_ok"] is not None]

    def mean_tokens(key: str) -> float | None:
        values = [r[key] for r in ok if r[key] is not None]
        return statistics.mean(values) if values else None

    return {
        "name": classifier.name,
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "handoff_acc": sum(r["handoff_ok"] for r in ok) / len(ok),
        "intent_acc": sum(r["intent_ok"] for r in intent_rows) / len(intent_rows),
        "confirmed_acc": sum(r["confirmed_ok"] for r in confirmed_rows)
        / len(confirmed_rows),
        "frustration_mae": statistics.mean(r["frustration_err"] for r in ok),
        "p50_ms": pct(latencies, 0.5) * 1000,
        "p95_ms": pct(latencies, 0.95) * 1000,
        "input_tokens": mean_tokens("input_tokens"),
        "output_tokens": mean_tokens("output_tokens"),
        "rows": rows,
    }


def print_summary(results: list[dict]) -> None:
    header = f"{'classifier':<32} {'n':>4} {'err':>4} {'handoff':>8} {'intent':>7} {'confirm':>8} {'frust MAE':>9} {'p50 ms':>7} {'p95 ms':>7} {'in tok':>7} {'out tok':>7}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if "n" not in r:
            print(f"{r['name']:<32} ERROR {r['error']}")
            continue
        tin = f"{r['input_tokens']:.0f}" if r["input_tokens"] is not None else "-"
        tout = f"{r['output_tokens']:.0f}" if r["output_tokens"] is not None else "-"
        print(
            f"{r['name']:<32} {r['n']:>4} {r['errors']:>4} {r['handoff_acc']:>8.0%} "
            f"{r['intent_acc']:>7.0%} {r['confirmed_acc']:>8.0%} {r['frustration_mae']:>9.2f} {r['p50_ms']:>7.0f} "
            f"{r['p95_ms']:>7.0f} {tin:>7} {tout:>7}"
        )
    print(
        f"\nhandoff: share of cases where (probability >= {HANDOFF_THRESHOLD}) matches the label."
        "\nintent: exact match, ambiguous cases skipped. frust MAE: mean distance from the 0-2 label."
        f"\nconfirm: share of readback cases where (probability >= {CONFIRMED_THRESHOLD}) matches the label."
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-models", default="openai/gpt-6-luna-decisions")
    parser.add_argument("--llms", default="google/gemma-4-31b-it")
    parser.add_argument("--cases", default=str(ROOT / "bench" / "cases.json"))
    parser.add_argument("--runs", type=int, default=1, help="passes over the case set")
    parser.add_argument("--verbose", action="store_true", help="print each case")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    classifiers = [
        DecisionClassifier(m) for m in args.decision_models.split(",") if m
    ] + [LLMClassifier(m) for m in args.llms.split(",") if m]

    results = []
    async with utils.http_context.open():
        for classifier in classifiers:
            print(f"\n{classifier.name}")
            results.append(
                await run_classifier(classifier, cases, args.runs, args.verbose)
            )

    print_summary(results)
    out = ROOT / "bench" / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    asyncio.run(main())
