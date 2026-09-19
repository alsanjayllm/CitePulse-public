"""The 5 v1 KPIs, curated from ABAEO's 71-KPI catalog for value proposition
and differentiation (see the CitePulse plan's "KPI bundle" section). Ids
match ABAEO's own numbering so a reader familiar with ABAEO can cross
reference directly; CitePulse itself never computes the other 66.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KPIDefinition:
    id: int
    name: str
    category: str  # foundation, citation, task_readiness
    unit: str
    higher_is_better: bool
    evidence_source: str  # crawl, ai_engine_prompt, task_readiness_trace


KPI_CATALOG: dict[int, KPIDefinition] = {
    # v2 "Foundation" KPI (cheap, lowest differentiation) -- added after
    # field evidence (an audit run against otterly.ai, a competitor AEO
    # tool) showed AI-crawler accessibility is a headline feature in this
    # competitive space.
    1: KPIDefinition(
        id=1,
        name="AI Crawl Accessibility",
        category="foundation",
        unit="score_0_to_3",
        higher_is_better=True,
        evidence_source="crawl",
    ),
    # v2 "Foundation" KPI, Phase 5 of the 18 Sept 2026 product-loop review
    # cycle -- the most-cited gap across all four field-review agents, and
    # the last remaining Foundation roadmap item.
    3: KPIDefinition(
        id=3,
        name="Schema Markup Coverage",
        category="foundation",
        unit="score_0_to_3",
        higher_is_better=True,
        evidence_source="crawl",
    ),
    46: KPIDefinition(
        id=46,
        name="llms.txt Readiness",
        category="foundation",
        unit="score_0_to_3",
        higher_is_better=True,
        evidence_source="crawl",
    ),
    22: KPIDefinition(
        id=22,
        name="Citation Rate",
        category="citation",
        unit="percent",
        higher_is_better=True,
        evidence_source="ai_engine_prompt",
    ),
    24: KPIDefinition(
        id=24,
        name="AI Share of Voice",
        category="citation",
        unit="score_0_to_100",
        higher_is_better=True,
        evidence_source="ai_engine_prompt",
    ),
    # Phase 4 (FR-4/FR-5): new, additive citation KPIs -- distinct IDs so
    # existing reports/tests/report consumers of #22/#24 never change shape.
    45: KPIDefinition(
        id=45,
        name="Citation Correctness Rate",
        category="citation",
        unit="percent",
        higher_is_better=True,
        evidence_source="ai_engine_prompt",
    ),
    62: KPIDefinition(
        id=62,
        name="AI Share of Voice (Weighted)",
        category="citation",
        unit="score_0_to_100",
        higher_is_better=True,
        evidence_source="ai_engine_prompt",
    ),
    48: KPIDefinition(
        id=48,
        name="Task Completion Success Rate",
        category="task_readiness",
        unit="percent",
        higher_is_better=True,
        evidence_source="task_readiness_trace",
    ),
    58: KPIDefinition(
        id=58,
        name="Interaction Readiness",
        category="task_readiness",
        unit="percent",
        higher_is_better=True,
        evidence_source="task_readiness_trace",
    ),
}
