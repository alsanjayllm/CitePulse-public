"""FR-1/FR-2 prompt-set management: read, import, and validate a site's
custom ``PromptItem`` corpus (the "prompt set" from the SRS). Managed via
the ``citepulse prompts validate/import`` CLI subcommands and the Streamlit
prompt-management page.

The corpus is versioned and soft-deleted via ``PromptItem.active``: importing
a new set deactivates the site's previously-active rows and inserts the new
ones as the active ``version``. Active rows are what
``citepulse.audit.run_audit`` feeds back into
``citation_rate.check_citation_rate(custom_prompts=...)`` when a site has
opted in; a site with no active prompts keeps using the built-in template
corpus (non-breaking). Validation is authoring-time only
(``citepulse.prompt_quality.validate_prompt_set``) -- it reports and never
mutates beyond the import/versioning described here.
"""

from __future__ import annotations

import csv
import io
import json
from uuid import UUID

from sqlmodel import Session, select

from citepulse.models import PromptItem
from citepulse.prompt_quality import PromptQualityReport, validate_prompt_set


def active_site_prompts(session: Session, site_id: UUID) -> list[PromptItem]:
    """Every active PromptItem for a site, ordered deterministically
    (by id) so the same corpus yields the same probe order. Empty list when
    the site has no active prompt set (i.e. it uses the built-in corpus)."""
    rows = session.exec(
        select(PromptItem)
        .where(PromptItem.site_id == site_id, PromptItem.active.is_(True))
        .order_by(PromptItem.id)
    ).all()
    return list(rows)


def current_prompt_version(session: Session, site_id: UUID) -> int:
    """The highest PromptItem.version for a site (0 when none exist) -- used
    so an import bumps version rather than reusing a number."""
    rows = session.exec(
        select(PromptItem).where(PromptItem.site_id == site_id)
    ).all()
    if not rows:
        return 0
    return max(r.version for r in rows)


def import_prompt_set(
    session: Session,
    site_id: UUID,
    items: list[dict],
) -> PromptQualityReport:
    """Replaces a site's active PromptItem corpus with a new version built
    from ``items`` (each a dict with ``text``, ``intent``, optional
    ``topic_cluster``/``job_to_be_done``/``difficulty``/
    ``expected_entities``/``expected_citation_domains``), deactivating the
    previously-active rows. Validates the newly-imported set and returns its
    prompt-quality report. Deactivation-then-insert and the report are
    committed atomically."""
    old = active_site_prompts(session, site_id)
    new_version = current_prompt_version(session, site_id) + 1

    for row in old:
        row.active = False

    prompts = []
    for item in items:
        prompts.append(
            PromptItem(
                site_id=site_id,
                text=(item.get("text") or "").strip(),
                intent=item.get("intent") or "",
                job_to_be_done=(item.get("job_to_be_done") or "").strip() or None,
                topic_cluster=(item.get("topic_cluster") or "").strip() or None,
                expected_entities=list(item.get("expected_entities") or []),
                expected_citation_domains=list(
                    item.get("expected_citation_domains") or []
                ),
                difficulty=(item.get("difficulty") or "").strip() or None,
                version=new_version,
                active=True,
            )
        )

    report = validate_prompt_set(prompts)
    session.add_all(prompts)
    session.commit()
    return report


def parse_prompt_file(path: str) -> list[dict]:
    """Parses a prompt-library file (`.json` or `.csv`) into a list of
    ``{text, intent, topic_cluster, ...}`` dicts. JSON accepts either a bare
    list of prompt definitions or an object with a ``prompts`` key (SRS FR-1
    shape). CSV expects columns ``text,intent[,topic_cluster,...]`` with a
    header row."""
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    if path.lower().endswith(".json"):
        data = json.loads(content)
        if isinstance(data, dict):
            items = data.get("prompts", [])
        else:
            items = data
        return [
            {k: (v if v is not None else "") for k, v in item.items()}
            for item in items
            if isinstance(item, dict)
        ]

    reader = csv.DictReader(io.StringIO(content))
    return [{k.strip(): (v if v is not None else "") for k, v in row.items()} for row in reader]


def validate_site_prompts(session: Session, site_id: UUID) -> PromptQualityReport:
    """Validates the site's currently-active prompt set (the plan's
    ``citepulse prompts validate`` path). A site with no active set returns
    an empty/blocking report noting that no corpus is authored yet."""
    prompts = active_site_prompts(session, site_id)
    return validate_prompt_set(prompts)
