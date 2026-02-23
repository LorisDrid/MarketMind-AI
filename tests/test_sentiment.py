"""
tests/test_sentiment.py — Integration tests for the Ollama sentiment analyser.

All three tests call the live local Ollama instance via ``get_sentiment``.
They are automatically **skipped** when Ollama is not reachable, so they
never block CI pipelines or machines without a GPU / running Ollama.

Run manually:
    pytest tests/test_sentiment.py -v

Expected runtime (RTX 5070, llama3 8B, warm model):
    ~3–6 s per test case
"""

from __future__ import annotations

import pytest

from sentiment_analyzer import _check_health, get_sentiment


# ---------------------------------------------------------------------------
# Module-level Ollama availability check
# ---------------------------------------------------------------------------
# Evaluated once at collection time; all tests are skipped in a single pass
# rather than failing on import or in each test body.

_OLLAMA_UP: bool = _check_health()

pytestmark = pytest.mark.skipif(
    not _OLLAMA_UP,
    reason="Ollama is not reachable — start the server with: ollama serve",
)


# ---------------------------------------------------------------------------
# Test fixtures / headline sets
# ---------------------------------------------------------------------------

# Unambiguously positive: blockbuster earnings, analyst upgrades, all-time highs
BULLISH_HEADLINES: list[str] = [
    "Company crushes Q4 earnings, stock surges 20% after massive beat",
    "Record-breaking revenue reported; profit forecast tripled by management",
    "Stock hits all-time high on blockbuster growth and strong forward guidance",
    "Analyst upgrades to Strong Buy: sees 60% upside in next 12 months",
    "Quarterly earnings shatter expectations; CEO raises full-year outlook",
]

# Unambiguously negative: bankruptcy, fraud, massive miss, regulatory fines
BEARISH_HEADLINES: list[str] = [
    "Company files for Chapter 11 bankruptcy amid catastrophic revenue collapse",
    "CEO resigns following massive accounting fraud scandal under SEC investigation",
    "Stock crashes 40% after earnings miss; full-year guidance withdrawn entirely",
    "Regulators impose record fine; multiple class-action lawsuits filed by investors",
    "Revenues plummet 65% year-over-year; 30% of workforce laid off immediately",
]

# Entirely irrelevant to financial performance or stock price
NEUTRAL_HEADLINES: list[str] = [
    "Company updates its internal expense reporting software interface",
    "Annual shareholder meeting rescheduled for next Tuesday afternoon",
    "New vending machine installed in the office break room on floor 3",
    "Standard quarterly filing submitted to the SEC on time as required",
    "Office lease renewed for another three years at unchanged current terms",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bullish_news() -> None:
    """Extremely positive headlines must return a score strictly above +0.5."""
    score = get_sentiment(BULLISH_HEADLINES, ticker="TEST")
    assert score > 0.5, (
        f"Expected bullish score > 0.5 for clearly positive headlines, "
        f"got {score:.4f}"
    )


def test_bearish_news() -> None:
    """Extremely negative headlines must return a score strictly below −0.5."""
    score = get_sentiment(BEARISH_HEADLINES, ticker="TEST")
    assert score < -0.5, (
        f"Expected bearish score < -0.5 for clearly negative headlines, "
        f"got {score:.4f}"
    )


def test_neutral_news() -> None:
    """Irrelevant, non-financial headlines must return a score within ±0.4.

    A looser bound (0.4) is used here because LLMs may assign a small
    positive/negative lean to company-related content even when it is
    operationally trivial.  The intent is to confirm the model does not
    produce a strong signal on noise.
    """
    score = get_sentiment(NEUTRAL_HEADLINES, ticker="TEST")
    assert abs(score) < 0.4, (
        f"Expected near-neutral score within ±0.4 for irrelevant headlines, "
        f"got {score:.4f}"
    )
