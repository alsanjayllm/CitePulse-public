from pathlib import Path

import yaml

_MAPPING_PATH = Path(__file__).parent / "config" / "standard_kpi_mapping.yaml"
_mapping: dict | None = None


def _load_mapping() -> dict:
    global _mapping
    if _mapping is None:
        with open(_MAPPING_PATH, encoding="utf-8") as f:
            _mapping = yaml.safe_load(f)
    return _mapping


def get_business_context(kpi_id: int) -> list[dict]:
    """Returns the illustrative standard-KPI mapping for one CitePulse KPI,
    e.g. [{"metric": "Market share (%)", "source": "altexsoft"}, ...].
    Empty list if no mapping is defined."""
    return _load_mapping().get(str(kpi_id), [])


def format_business_context(context: list[dict]) -> str:
    """One shared rendering of get_business_context()'s output, used by
    both the Markdown report and the Streamlit UI so the two never drift
    on how a mapping is displayed."""
    return "; ".join(f"{c['metric']} ({c['source']})" for c in context)
