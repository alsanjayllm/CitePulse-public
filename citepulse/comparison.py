"""Comparison mode: run the same audit once per model so a user can
eyeball which model evaluates citation/task-completion more reliably.
Originally Ollama-only (Track C); now also reaches OpenRouter (cloud)
models via citepulse.ai_engines.provider's `openrouter:`-prefixed model
strings -- run_comparison() itself doesn't know or care which provider a
given model string resolves to, since that dispatch lives entirely in
provider.py and everything it's called by (see that module's docstring
for the full provider-dispatch story). A 3-model UI comparison
(citepulse/ui/pages/compare.py) additionally derives a 4th, consolidated
view (citepulse.comparison_consolidation.consolidate_runs) -- computed
fresh at render time from 3 real runs, never a persisted 4th "virtual"
run.

`api_keys` (new): an OpenRouter API key is Streamlit session-state-only,
per the locked-in "no account" posture -- it is passed down through
run_audit() -> each KPI runner -> citepulse.ai_engines.provider as a
plain kwarg, exactly the same pattern already established for `model`,
and is NEVER written to Settings/.env/the DB anywhere in this call chain.

Deliberately sequential, never parallel: a single local Ollama instance
and a single Playwright-driven task-readiness harness per run would only
contend with each other if run concurrently -- multiple audits at once
would just make all of them slower, not faster, while adding real
complexity (thread safety of the task-readiness trace cache, GPU/CPU
contention on the Ollama side) for no benefit. This applies even less to
an OpenRouter-backed run (a stateless HTTP call, no local contention) --
but sequencing is still the simplest correct behavior when a comparison
mixes providers, so no partial-parallelism special case is added here.
"""

from collections.abc import Callable

from sqlmodel import Session

from citepulse.audit import run_audit
from citepulse.models import AuditRun


def run_comparison(
    session: Session,
    url: str,
    models: list[str],
    kpi_ids: list[int] | None = None,
    on_progress: Callable[[str], None] | None = None,
    api_keys: dict[str, str] | None = None,
) -> list[AuditRun]:
    """Runs `citepulse.audit.run_audit(session, url, model=m, kpi_ids=kpi_ids)`
    once for each `m` in `models`, in order, returning the resulting
    AuditRun list in the same order. Each call creates its own independent
    AuditRun row (with its own resolved AuditRun.model) -- there is no
    shared state between the two runs beyond the site itself, so either
    run's failure (a KPI-runner exception, an unreachable model, an
    unknown `kpi_ids` entry) surfaces exactly the way a single
    `run_audit()` call already does, uncaught.

    `kpi_ids` (same convention as `run_audit()`'s own parameter -- `None`
    means "all") is threaded identically into every model's `run_audit()`
    call: comparing different KPI subsets across models would be
    meaningless, so there is deliberately no per-model override here.
    Omitted entirely (the default), `run_audit()` is called with no
    `kpi_ids` kwarg at all -- same "unaffected when omitted" guarantee as
    `on_progress` below.

    `api_keys` (new): keyed by model string (not one flat key), so a
    3-model comparison mixing Ollama and OpenRouter models works with
    zero special-casing -- each model's own `run_audit()` call gets
    `api_keys.get(model_name)` (None for a model absent from the dict,
    e.g. every Ollama model when only an OpenRouter slot's key was
    entered). Omitted entirely (the default, `None`), no `api_key` kwarg
    is passed to `run_audit()` at all -- existing Ollama-only callers are
    completely unaffected.

    `on_progress`, when given, is wrapped per model so each forwarded
    message is prefixed with which model produced it (e.g. "Model 1/2
    (llama3.1:8b): Running Citation Rate...") -- callers otherwise have no
    way to tell which of the sequential run_audit() calls a message came
    from. Omitted entirely, `run_audit()` is called exactly as before (no
    on_progress kwarg at all), so existing callers/tests are unaffected."""
    total = len(models)
    runs = []
    for index, model_name in enumerate(models, start=1):
        kwargs = {"model": model_name}
        if kpi_ids is not None:
            kwargs["kpi_ids"] = kpi_ids
        if api_keys is not None:
            kwargs["api_key"] = api_keys.get(model_name)
        if on_progress is not None:

            def _model_progress(
                message: str, _model=model_name, _index=index, _total=total
            ) -> None:
                on_progress(f"Model {_index}/{_total} ({_model}): {message}")

            kwargs["on_progress"] = _model_progress
        runs.append(run_audit(session, url, **kwargs))
    return runs
