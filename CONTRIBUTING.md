# Contributing to CitePulse

Thanks for taking a look. CitePulse is young — there's no separate
architecture doc yet; the source is the documentation. A few pointers
before you dive in:

- `citepulse/kpis/kpi_N.py` — one file per KPI. Each turns evidence into a
  `KPIResult` and an optional `Finding`.
- A `Finding` is only ever created for a real, detected gap — never a
  placeholder, and never for a KPI already at its best band.
- A KPI value is never fabricated: unmeasurable/unavailable renders as
  `None`/"not determined," never a silent `0`.
- Remediation text is generated once at audit-run time and persisted on
  the `Finding` row — reopening a past run must never regenerate it.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest
```

## Adding a new KPI

Follow the existing pattern used by any `citepulse/kpis/kpi_N.py` module:
an evidence-gathering function (in `citepulse/crawler/` or
`citepulse/ai_engines/`, depending on what it needs) → a `kpis/kpi_N.py`
module that scores it into a `KPIResult`/`Finding` → remediation templates
in `citepulse/config/remediation.yaml` → a row in
`citepulse/config/standard_kpi_mapping.yaml` → tests → wiring into
`citepulse/kpi_catalog.py` and the CLI. Look at an existing KPI (e.g.
`citepulse/kpis/kpi_46.py`) as the reference implementation.

## Before opening a PR

- `pytest` passes
- No secrets, API keys, or tokens in your diff (a pre-commit hook checks
  for this, but review your own diff too)
- New behavior gets a test

## License

By contributing, you agree your contribution is licensed under this
project's [MIT license](LICENSE).
