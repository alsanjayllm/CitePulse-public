"""Shared sample-size-aware confidence math -- a Wilson score interval,
appropriate for small-n binomial rates (e.g. a dozen citation-check
prompts, or a handful of task-readiness runs) where a naive normal
approximation can produce a nonsensical interval (below 0 or above 1).
Used by citepulse.kpis.kpi_48/kpi_58 (Phase 1), and by kpi_22/kpi_24
(Phase 3) -- replacing every ad hoc "medium if capped else high"-style
heuristic with one real statistical function. kpi_22's value is a plain
binomial rate, so wilson_confidence's interval genuinely bounds it
(confidence_interval_low/high populated); kpi_24's value is a position/
frequency-weighted share, not a plain proportion, so it's applied there
only to a proxy binomial for the confidence *label* -- see kpi_24.py's
own comment for why confidence_interval_low/high are deliberately left
unset in that case.

Never fabricates, never raises: an n<=0 (or an otherwise degenerate)
input returns a zero-width interval at 0.0 with the "low" label rather
than dividing by zero or propagating an exception -- same defensive
posture as remediation.py's `_SafeDict` and ai_engines/ollama.py's
always-return-never-raise contract.
"""

import math


def wilson_confidence(successes: int, n: int) -> tuple[float, float, str]:
    """Returns (low, high, label) -- low/high are fractions in [0.0, 1.0]
    (not percentages), and label is one of "high"/"medium"/"low" derived
    from both the sample size and the interval's width:
      - "high"   if n >= 20 and (high - low) <= 0.15
      - "medium" if n >= 8  and (high - low) <= 0.35
      - "low"    otherwise (including any degenerate/unmeasurable input)
    """
    if n <= 0 or successes < 0 or successes > n:
        return (0.0, 0.0, "low")

    p = successes / n
    z = 1.96
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2)) / denom

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    width = high - low

    if n >= 20 and width <= 0.15:
        label = "high"
    elif n >= 8 and width <= 0.35:
        label = "medium"
    else:
        label = "low"

    return (low, high, label)
