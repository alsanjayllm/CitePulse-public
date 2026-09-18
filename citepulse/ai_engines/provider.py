"""Single dispatch point between CitePulse's two LLM providers -- local
Ollama (the only provider until now) and OpenRouter (cloud), added so a
3-way model comparison can mix a cloud model into the same audit
pipeline everything else already runs against. Not an ABC/protocol
layer -- that isn't this codebase's style (see task_generator.py's
"no engine-abstraction layer" precedent) -- just two flat functions matching
citepulse.ai_engines.ollama's exact shape, so every existing call site
only needs an import swap plus an added `api_key` kwarg (see this
package's other modules' own docstrings for exactly which ones).

Model-string convention: a model name prefixed with `openrouter:` (e.g.
"openrouter:anthropic/claude-3-haiku") selects OpenRouter; anything else,
including `None`, is Ollama, completely unchanged from before this
module existed. `_split_model()` is the ONE place in CitePulse that
parses this prefix -- every other module (citation_rate.py,
task_generator.py, harness.py, business_narrative.py, company_profile.py,
every kpi_N.py) calls only this module's ask()/ask_with_retry(), never
ollama.py or openrouter.py directly, so neither downstream module ever
sees the `openrouter:` prefix still attached.

Free-tier rotation: OpenRouter's `:free` models share a small upstream
capacity pool and get rate-limited after only a handful of burst calls
(no Retry-After header sent). CitePulse makes many back-to-back LLM
calls per run (12 citation probes + extra classifiers + task generation
+ narrative), which bursts past any single free model's window. When a
`:free` model is selected and settings.openrouter_model_rotation_enabled
is on (the default), ask()/ask_with_retry() therefore round-robin each
call across ALL free entries in openrouter_catalog.yaml via
`_rotate_free_model()`, so the workload spreads across the pool instead
of exhausting one shared free pool. The run stays labeled with (and
compared by) the model the user picked -- rotation is purely an internal
distribution mechanism; each result's raw_data still records the exact
model that answered. Paid OpenRouter models and Ollama never rotate.
"""

from citepulse.ai_engines import ollama, openrouter

_OPENROUTER_PREFIX = "openrouter:"

# Round-robin cursor for free-tier rotation (see module docstring). A
# plain module-level counter guarded by a lock is enough -- requests are
# short-lived and the counter is never held across an await. Kept here
# (not on the Settings object) because rotation is a per-process call
# distribution concern, not a persisted configuration value.
_free_cursor = 0
_free_lock = None


def _split_model(model: str | None) -> tuple[str, str | None]:
    """Returns (provider, bare_model) where provider is "openrouter" or
    "ollama". `bare_model` is `model` with the `openrouter:` prefix
    stripped (None stays None for the Ollama case, matching ollama.ask's
    own "None means use the configured default" convention -- OpenRouter
    has no such default, so a bare `openrouter:` with nothing after the
    prefix yields bare_model="", which openrouter.ask() already treats as
    an immediate unavailable result)."""
    if model is not None and model.startswith(_OPENROUTER_PREFIX):
        return "openrouter", model[len(_OPENROUTER_PREFIX) :]
    return "ollama", model


def _free_rotation_pool() -> list[str] | None:
    """The free-tier rotation pool: every `openrouter:`-prefixed name in
    config/openrouter_catalog.yaml whose bucket is "free", in catalog
    order. Returns None when there are fewer than 2 free entries -- with
    only one (or zero) free model there's nothing meaningful to rotate
    across, so rotation is disabled and the model is always called as
    picked. Reads the catalog lazily (not at import) via
    citepulse.model_recommender.list_openrouter_models() -- deferred so
    provider.py has no import-time dependency on that module (which drags
    in psutil + ollama's tag-listing), and so an unreadable/missing
    catalog degrades to "no rotation" instead of raising."""
    try:
        from citepulse.model_recommender import list_openrouter_models

        pool = [
            opt.name
            for opt in list_openrouter_models()
            if opt.name.startswith(_OPENROUTER_PREFIX) and opt.bucket == "free"
        ]
    except Exception:
        return None
    if len(pool) < 2:
        return None
    return pool


def _rotate_free_model(model: str | None) -> str | None:
    """Round-robin free-tier rotation, operating on a *bare* OpenRouter id
    (no `openrouter:` prefix -- callers already stripped it via
    _split_model before calling this). If `model` is a `:free` OpenRouter
    model and rotation is enabled (settings.openrouter_model_rotation_enabled
    defaults True) and the catalog has >= 2 free entries, returns the next
    distinct free model id in the pool (bare, prefix stripped) for this
    call; otherwise returns `model` unchanged.

    Each call advances a process-wide cursor so a run's back-to-back
    probe/classifier/narrative calls spread across the whole free pool
    instead of hammering one shared upstream free pool (see module
    docstring). The caller's selected `model` stays the run's label; this
    only picks *which* free endpoint answers this particular call."""
    if not model or not model.endswith(":free"):
        return model
    try:
        from citepulse.settings import get_settings

        if not get_settings().openrouter_model_rotation_enabled:
            return model
    except Exception:
        return model
    pool = _free_rotation_pool()
    if pool is None:
        return model

    global _free_cursor, _free_lock
    if _free_lock is None:
        import threading

        _free_lock = threading.Lock()
    # Advance the cursor and pick a candidate, scanning at most a full
    # cycle of the pool so we can't spin forever. Prefer a model other
    # than the one the user picked so a burst doesn't run the selected
    # model back-to-back into its own upstream rate limit; if a full pass
    # only ever offers the selected model (degenerate pool), use it.
    candidate = None
    for _ in range(len(pool)):
        with _free_lock:
            full = pool[_free_cursor % len(pool)]
            _free_cursor += 1
        candidate = full[len(_OPENROUTER_PREFIX) :]
        if candidate != model:
            return candidate
    # Every slot held the selected model -- nothing else to rotate onto.
    return candidate


def ask(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
) -> dict:
    """Dispatches to citepulse.ai_engines.ollama.ask() or
    citepulse.ai_engines.openrouter.ask() based on `model`'s
    `openrouter:` prefix (see _split_model). Same
    {"available", "text", "model", "raw_data"} return shape and
    never-raise contract either module already provides -- this function
    adds no behavior of its own beyond routing (plus, for a `:free`
    OpenRouter model, free-tier rotation -- see _rotate_free_model)."""
    provider, bare_model = _split_model(model)
    if provider == "openrouter":
        return openrouter.ask(
            prompt,
            context=context,
            system=system,
            model=_rotate_free_model(bare_model),
            timeout=timeout,
            api_key=api_key,
        )
    return ollama.ask(
        prompt, context=context, system=system, model=bare_model, timeout=timeout
    )


def ask_with_retry(
    prompt: str,
    *,
    context: str | None = None,
    system: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    api_key: str | None = None,
) -> dict:
    """Dispatches to the matching provider's ask_with_retry() -- same
    routing as ask() above, including free-tier rotation (the retries of
    a single logical call reuse the model this call was rotated onto, so
    ask_with_retry's pacing/backoff stays consistent per logical call)."""
    provider, bare_model = _split_model(model)
    if provider == "openrouter":
        return openrouter.ask_with_retry(
            prompt,
            context=context,
            system=system,
            model=_rotate_free_model(bare_model),
            timeout=timeout,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            api_key=api_key,
        )
    return ollama.ask_with_retry(
        prompt,
        context=context,
        system=system,
        model=bare_model,
        timeout=timeout,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
