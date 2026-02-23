"""
scripts/benchmark_sentiment.py — Sentiment model accuracy benchmark.

Loads hand-labelled financial headlines from ``tests/sentiment_benchmark.json``,
runs each headline through the local Ollama sentiment analyser, then reports:

  * Per-headline comparison (model score vs. human label)
  * Mean Squared Error  (MSE)   — measures numeric accuracy
  * Sign Accuracy (%)          — fraction where model and human agree on direction
                                  (both positive, both negative, or both neutral)

The neutral band is ±0.15: scores within this range are treated as "neutral"
for the sign-accuracy calculation (avoids penalising tiny disagreements on
near-zero labels).

Usage
-----
    python scripts/benchmark_sentiment.py
    python scripts/benchmark_sentiment.py --json tests/sentiment_benchmark.json
    python scripts/benchmark_sentiment.py --verbose   # show per-headline detail

Exit codes
----------
  0  Benchmark completed (results printed regardless of accuracy)
  1  Ollama unreachable or benchmark JSON not found
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

# Ensure the project root is importable when running from the scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sentiment_analyzer import _check_health, get_sentiment  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_BENCHMARK_PATH = _PROJECT_ROOT / "tests" / "sentiment_benchmark.json"

# Scores in (-NEUTRAL_BAND, +NEUTRAL_BAND) are treated as "neutral" for sign
# accuracy purposes.  This avoids penalising the model for disagreements like
# human_label=0.0 vs model_score=0.05.
NEUTRAL_BAND: float = 0.15

# ── Helpers ───────────────────────────────────────────────────────────────────


def _sign_bucket(score: float) -> str:
    """Map a score to 'positive', 'neutral', or 'negative'."""
    if score > NEUTRAL_BAND:
        return "positive"
    if score < -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def _bar(value: float, width: int = 20) -> str:
    """ASCII bar: value in [-1, 1] → string like '─────O───────────'."""
    clamped  = max(-1.0, min(1.0, value))
    position = int((clamped + 1) / 2 * width)
    return "─" * position + "●" + "─" * (width - position)


# ── Main benchmark logic ──────────────────────────────────────────────────────


def run_benchmark(
    benchmark_path: Path,
    verbose: bool = False,
) -> dict:
    """Run the full benchmark and return a result summary dict.

    Returns:
        dict with keys: ``mse``, ``sign_accuracy``, ``n``, ``n_correct_sign``,
        ``results`` (list of per-headline dicts).
    """
    # ── Load benchmark data ──────────────────────────────────────────────────
    if not benchmark_path.exists():
        print(f"[ERROR] Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    with benchmark_path.open("r", encoding="utf-8") as fh:
        entries: list[dict] = json.load(fh)

    if not entries:
        print("[ERROR] Benchmark file is empty.", file=sys.stderr)
        sys.exit(1)

    SEP  = "=" * 70
    SEP2 = "-" * 70

    print(f"\n{SEP}")
    print(f"  MarketMind AI — Sentiment Benchmark")
    print(f"  Source : {benchmark_path.relative_to(_PROJECT_ROOT)}")
    print(f"  Items  : {len(entries)}")
    print(SEP)

    # ── Ollama health check ──────────────────────────────────────────────────
    if not _check_health():
        print(
            "\n[ERROR] Ollama is not reachable.\n"
            "  Start it with:  ollama serve\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n  Running... (first call may take up to 60 s for model load)\n")

    # ── Evaluate each headline ───────────────────────────────────────────────
    per_headline: list[dict] = []
    squared_errors: list[float] = []
    sign_matches: list[bool] = []

    for idx, entry in enumerate(entries, 1):
        ticker:      str   = entry.get("ticker", "")
        headline:    str   = entry["headline"]
        human_label: float = float(entry["human_label"])

        model_score: float = get_sentiment([headline], ticker=ticker)
        error       = model_score - human_label
        sq_err      = error ** 2

        human_bucket = _sign_bucket(human_label)
        model_bucket = _sign_bucket(model_score)
        sign_match   = (human_bucket == model_bucket)

        squared_errors.append(sq_err)
        sign_matches.append(sign_match)

        per_headline.append({
            "idx":          idx,
            "ticker":       ticker,
            "headline":     headline,
            "human_label":  human_label,
            "model_score":  model_score,
            "error":        round(error, 4),
            "sq_error":     round(sq_err, 4),
            "sign_match":   sign_match,
        })

        if verbose:
            match_icon = "✓" if sign_match else "✗"
            print(f"  [{idx:>2}/{len(entries)}] {match_icon}  {ticker:<8}  "
                  f"human={human_label:+.2f}  model={model_score:+.2f}  "
                  f"err={error:+.3f}")
            print(f"         [{_bar(human_label)}] human")
            print(f"         [{_bar(model_score)}] model")
            short = headline[:62] + "…" if len(headline) > 63 else headline
            print(f"         \"{short}\"")
            print()

    # ── Aggregate metrics ────────────────────────────────────────────────────
    n             = len(squared_errors)
    mse           = sum(squared_errors) / n
    rmse          = math.sqrt(mse)
    n_correct     = sum(sign_matches)
    sign_accuracy = n_correct / n * 100

    # ── Results table ────────────────────────────────────────────────────────
    if not verbose:
        # Show compact per-headline table even in non-verbose mode
        print(f"  {'#':>2}  {'Ticker':<8}  {'Human':>6}  {'Model':>6}  {'Err':>6}  {'✓'}")
        print(f"  {SEP2}")
        for r in per_headline:
            icon = "✓" if r["sign_match"] else "✗"
            print(
                f"  {r['idx']:>2}  {r['ticker']:<8}  "
                f"{r['human_label']:>+6.2f}  {r['model_score']:>+6.2f}  "
                f"{r['error']:>+6.3f}  {icon}"
            )
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(SEP)
    print(f"  BENCHMARK RESULTS ({n} headlines)")
    print(SEP2)
    print(f"  MSE            : {mse:.4f}")
    print(f"  RMSE           : {rmse:.4f}  (avg absolute deviation)")
    print(f"  Sign Accuracy  : {sign_accuracy:.1f}%  ({n_correct}/{n} correct direction)")
    print(f"  Neutral band   : ±{NEUTRAL_BAND:.2f}  (scores in this range = neutral)")
    print(SEP)

    # ── Per-ticker breakdown ─────────────────────────────────────────────────
    tickers_seen: list[str] = []
    for r in per_headline:
        if r["ticker"] not in tickers_seen:
            tickers_seen.append(r["ticker"])

    if len(tickers_seen) > 1:
        print("\n  Per-ticker breakdown:")
        print(f"  {SEP2}")
        for t in tickers_seen:
            t_rows = [r for r in per_headline if r["ticker"] == t]
            t_mse  = sum(r["sq_error"] for r in t_rows) / len(t_rows)
            t_acc  = sum(r["sign_match"] for r in t_rows) / len(t_rows) * 100
            print(
                f"  {t:<8}  n={len(t_rows):>2}  "
                f"MSE={t_mse:.3f}  Sign={t_acc:.0f}%"
            )
        print()

    return {
        "mse":            round(mse, 4),
        "rmse":           round(rmse, 4),
        "sign_accuracy":  round(sign_accuracy, 2),
        "n":              n,
        "n_correct_sign": n_correct,
        "results":        per_headline,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,      # suppress sentiment_analyzer INFO logs
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MarketMind AI — Sentiment benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-headline ASCII gauge bars",
    )
    args = parser.parse_args()

    run_benchmark(benchmark_path=args.json, verbose=args.verbose)
    sys.exit(0)
