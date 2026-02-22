"""
sentiment_analyzer.py — Local LLM sentiment scoring via Ollama.

Sends batched financial headlines to a locally-running Ollama instance and
returns a single score in [-1.0, 1.0] representing the aggregate sentiment.

Score convention:
  -1.0  →  Extremely bearish
   0.0  →  Neutral  (also returned on any failure)
  +1.0  →  Extremely bullish

Performance (RTX 5070, llama3 8B):
  Cold start (model load)  : ~40–60 s  (one-time per Ollama session)
  Warm call (10 headlines) :  ~2–4 s   (acceptable for backtesting loops)

Usage:
    from sentiment_analyzer import get_sentiment
    score = get_sentiment(["Nvidia beats Q4 estimates", ...], ticker="NVDA")
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL:   str = "llama3.1"   # pull with: ollama pull llama3.1
DEFAULT_TIMEOUT: int = 90           # seconds — covers cold-start load time

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"


# ── Internal helpers ───────────────────────────────────────────────────────

def _check_health(timeout: int = 5) -> bool:
    """Return True if Ollama is reachable at OLLAMA_BASE_URL."""
    try:
        r = requests.get(_TAGS_URL, timeout=timeout)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> list[str]:
    """Return the names of all models currently pulled in Ollama.

    Returns an empty list if Ollama is unreachable.
    """
    try:
        r = requests.get(_TAGS_URL, timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []


def _resolve_model(requested: str) -> Optional[str]:
    """Return *requested* if it exists in Ollama, else the first available model.

    Logs a warning when falling back so the caller knows.
    Returns None if no models are pulled at all.
    """
    available = list_available_models()
    if not available:
        return None

    # Exact match first, then prefix match (e.g. "llama3.1" → "llama3.1:latest")
    for name in available:
        if name == requested or name.startswith(requested.split(":")[0]):
            if name != requested:
                logger.debug("Model '%s' matched as '%s'.", requested, name)
            return name

    # Fallback to first available
    fallback = available[0]
    logger.warning(
        "Requested model '%s' not found in Ollama. "
        "Falling back to '%s'. Available: %s",
        requested, fallback, available,
    )
    return fallback


def _build_prompt(headlines: list[str], ticker: str) -> str:
    """Compose the single batched prompt sent to the LLM."""
    numbered = "\n".join(f'{i + 1}. "{h}"' for i, h in enumerate(headlines))
    ctx = f" for {ticker}" if ticker else ""
    return (
        f"You are a financial market analyst. "
        f"Analyze the following news headlines{ctx} "
        f"and determine the overall market sentiment for the stock's short-term price action.\n\n"
        f"Headlines:\n{numbered}\n\n"
        f"Respond ONLY with a JSON object in this exact format — no explanation, no extra text:\n"
        f'{{\"score\": <float between -1.0 and 1.0>}}\n'
        f"Where -1.0 is extremely bearish and 1.0 is extremely bullish."
    )


def _parse_score(raw: str) -> Optional[float]:
    """Extract a float score from the model's raw text response.

    Parsing strategy (in order):
      1. Strict ``json.loads()``
      2. Regex on ``"score": <number>``
      3. First float in [-1, 1] found anywhere in the text

    Returns None when all strategies fail.
    """
    # 1. Strict JSON
    try:
        data = json.loads(raw)
        for key in ("score", "Score", "sentiment", "Sentiment"):
            if key in data:
                return max(-1.0, min(1.0, float(data[key])))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Regex: "score": 0.75
    m = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if m:
        try:
            return max(-1.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    # 3. Any float/int that fits in [-1, 1]
    for token in re.findall(r'-?\d+(?:\.\d+)?', raw):
        try:
            val = float(token)
            if -1.0 <= val <= 1.0:
                return val
        except ValueError:
            continue

    return None


def _interpret(score: float) -> str:
    """Human-readable label for a sentiment score."""
    if   score >=  0.6: return "Strong Bullish  [+++]"
    elif score >=  0.2: return "Mild Bullish    [+  ]"
    elif score >= -0.2: return "Neutral         [ 0 ]"
    elif score >= -0.6: return "Mild Bearish    [-  ]"
    else:               return "Strong Bearish  [---]"


# ── Public API ─────────────────────────────────────────────────────────────

def get_sentiment(
    text_list: list[str],
    ticker:    str = "",
    model:     str = DEFAULT_MODEL,
    timeout:   int = DEFAULT_TIMEOUT,
) -> float:
    """Analyze a batch of news headlines and return a single sentiment score.

    All headlines are sent in **one** prompt to minimise LLM round-trips,
    keeping the function practical inside backtesting time-step loops.

    Args:
        text_list: News headline strings to analyse.
        ticker:    Stock symbol used for context in the prompt (e.g. ``"NVDA"``).
        model:     Ollama model name.  Resolved automatically if not pulled.
        timeout:   HTTP request timeout in seconds.
                   Default (90 s) covers a cold-start model load on first call.

    Returns:
        ``float`` in ``[-1.0, 1.0]``.
        Returns ``0.0`` (neutral) on any error so callers never crash.
    """
    if not text_list:
        logger.warning("get_sentiment: empty headline list — returning 0.0")
        return 0.0

    # ── Ollama health ────────────────────────────────────────────────────
    if not _check_health():
        logger.warning(
            "Ollama is not reachable at %s. "
            "Start it with: `ollama serve`  — returning neutral 0.0",
            OLLAMA_BASE_URL,
        )
        return 0.0

    # ── Model resolution ─────────────────────────────────────────────────
    resolved_model = _resolve_model(model)
    if resolved_model is None:
        logger.warning(
            "No models pulled in Ollama. Run `ollama pull llama3.1` — returning 0.0"
        )
        return 0.0

    # ── Build & send request ─────────────────────────────────────────────
    prompt  = _build_prompt(text_list, ticker)
    payload = {
        "model":   resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":  False,
        "format":  "json",          # Ollama native JSON mode — forces valid JSON output
        "options": {"temperature": 0.0},  # deterministic scoring
    }

    logger.info(
        "Ollama ← %d headline(s) | ticker=%s | model=%s | timeout=%ds",
        len(text_list), ticker or "N/A", resolved_model, timeout,
    )

    t0 = time.perf_counter()
    try:
        response = requests.post(_CHAT_URL, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama connection refused — returning 0.0")
        return 0.0
    except requests.exceptions.Timeout:
        logger.warning(
            "Ollama timed out after %ds (ticker=%s) — returning 0.0",
            timeout, ticker or "N/A",
        )
        return 0.0
    except requests.exceptions.HTTPError as exc:
        logger.error("Ollama HTTP error: %s — returning 0.0", exc)
        return 0.0

    elapsed = time.perf_counter() - t0

    # ── Parse response ───────────────────────────────────────────────────
    try:
        content: str = response.json()["message"]["content"]
    except (KeyError, ValueError) as exc:
        logger.error("Malformed Ollama response (%s) — returning 0.0", exc)
        return 0.0

    logger.debug("Ollama raw → %s", content)

    score = _parse_score(content)
    if score is None:
        logger.warning(
            "Could not extract score from response %r — returning 0.0",
            content[:120],
        )
        return 0.0

    logger.info(
        "Ollama → score=%.4f | %s | ticker=%s | %.2fs",
        score, _interpret(score), ticker or "N/A", elapsed,
    )
    return score


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    SEP   = "=" * 60
    TICKER = "NVDA"

    print(f"\n{SEP}")
    print(f"  MarketMind AI - Sentiment Analyzer Smoke Test")
    print(SEP)

    # ── Ollama status ───────────────────────────────────────────────────
    print("\n[Ollama status]")
    if _check_health():
        models = list_available_models()
        print(f"  OK  - Reachable at {OLLAMA_BASE_URL}")
        print(f"  Models pulled : {models}")
    else:
        print(f"  FAIL - Ollama not reachable at {OLLAMA_BASE_URL}")
        print("  -> Start with: ollama serve")
        raise SystemExit(1)

    # ── Load cached headlines ───────────────────────────────────────────
    cache_path = pathlib.Path("data/cache/NVDA_news.json")
    if not cache_path.exists():
        print(f"\nWARN: Cache not found at {cache_path}.")
        print("  -> Run: python data_manager.py   to populate it.")
        raise SystemExit(1)

    headlines_raw: list[dict] = json.loads(cache_path.read_text("utf-8"))
    titles: list[str] = [item["title"] for item in headlines_raw if item.get("title")]

    print(f"\n[Headlines from cache - {len(titles)} total]")
    for i, title in enumerate(titles, 1):
        print(f"  {i:>2}. {title}")

    # ── Run sentiment analysis ──────────────────────────────────────────
    print(f"\n[Sending to Ollama — model: {DEFAULT_MODEL}]")
    print("  Note: first call may take 40-60 s while the model loads.\n")

    t_start = time.perf_counter()
    score   = get_sentiment(titles, ticker=TICKER)
    t_total = time.perf_counter() - t_start

    # ── Results ─────────────────────────────────────────────────────────
    bar_len  = 40
    bar_pos  = int((score + 1) / 2 * bar_len)
    bar      = "-" * bar_pos + "O" + "-" * (bar_len - bar_pos)

    print(f"\n{'-'*60}")
    print(f"  Ticker   : {TICKER}")
    print(f"  Score    : {score:+.4f}   ({_interpret(score)})")
    print(f"  Gauge    : [{bar}]")
    print(f"             Bearish (-1)             Bullish (+1)")
    print(f"  Elapsed  : {t_total:.2f} s")
    print(f"  Headlines: {len(titles)}")
    print(f"{'-'*60}\n")
