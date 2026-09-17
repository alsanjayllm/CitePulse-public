from pathlib import Path
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    perplexity_api_key: str | None = None

    # Optional search fallback keys -- used only if DuckDuckGo/Google News
    # return nothing. Base product never requires any of these.
    serper_api_key: str | None = None
    tavily_api_key: str | None = None
    bing_api_key: str | None = None

    remediation_llm_polish: bool = False

    # Local file logging only -- see citepulse/logging_setup.py; never
    # sent anywhere.
    log_level: str = "INFO"

    citepulse_data_dir: str = "~/.citepulse"
    retention_keep_last_runs: int = 20
    retention_keep_days: int = 90

    max_sites: int = 10

    # Citation Rate / AI Share of Voice (#22/#24) -- see
    # citepulse/ai_engines/citation_rate.py for the six-segment prompt
    # corpus this bounds. The built-in corpus has up to 3 templates per
    # segment (18 prompts total across category discovery/capability/
    # comparison/purchase/implementation/brand navigation). FR-2.5 moved
    # this from a fixed 12 to a tiered bound (`minimal` = 30, `production`
    # = 50) aligned with prompt_quality_tier below: the default of 30
    # still caps at the 18-prompt built-in corpus, and is an explicit
    # per-run execution/latency knob -- each prompt is a real web search
    # plus an LLM call. A site that has imported a custom PromptItem
    # corpus (via `citepulse prompts import`) runs those validated prompts
    # instead, still respecting an explicit num_prompts cap when given.
    citation_rate_max_prompts: int = 30

    # FR-2/FR-2.5 prompt-quality tier governing how large a validated
    # prompt set must be for statistical reliability. `minimal` requires
    # 30 total prompts (the default, matching the 30 above); `production`
    # requires 50+. Used by citepulse.prompt_quality.validate_prompt_set()
    # -- validation only warns/rejects at authoring time; downstream KPIs
    # render None/low-confidence for a corpus that's still too small,
    # never a fabricated number.
    prompt_quality_tier: str = "minimal"

    # FR-2.5: minimum prompts per major topic cluster for statistical
    # reliability. validate_prompt_set() warns when any cluster falls below
    # this (20 from the SRS), alongside the tier's total-prompt floor.
    prompt_quality_min_per_cluster: int = 20

    # Gates the 4 extra AI-visibility metrics (mention_rate,
    # recommendation_rate, citation_quality_score, message_accuracy) added
    # alongside citation_rate/share_of_voice -- see
    # citepulse/ai_engines/citation_rate.py's check_citation_rate() for
    # what each one measures. On by default; setting this False skips all
    # four entirely (zero extra Ollama calls beyond the base corpus) and
    # the evidence dict records extra_metrics_enabled=False so the report
    # can say "not computed" rather than rendering an absent value as if
    # it were tested-and-unavailable.
    citation_rate_extra_metrics_enabled: bool = True

    # citepulse.competitor_discovery.discover_competitors() -- a hard kill
    # switch for automated competitor discovery (2 web searches + 1 LLM
    # call), not a per-audit setting: discovery is always explicitly
    # invoked (the CLI's `citepulse competitor discover` subcommand, or a
    # UI button), never run automatically inside `citepulse audit`, so the
    # cost is already opt-in by construction. On by default; set False for
    # an environment that wants to disable outbound discovery calls
    # entirely -- discover_competitors() checks this first and returns []
    # with no search/LLM calls made when off.
    competitor_discovery_enabled: bool = True

    # Free-tier OpenRouter models (:free) share a small upstream capacity
    # pool and get rate-limited after only a handful of burst calls (no
    # Retry-After header is sent), but CitePulse makes many back-to-back
    # LLM calls per run (12 citation probes + extra classifiers + task
    # generation + narrative). When True (default), a run that selects a
    # :free OpenRouter model therefore round-robins its calls across ALL
    # free entries in openrouter_catalog.yaml (the free rotation pool),
    # paced to stay under the per-minute cap -- see
    # citepulse.ai_engines.provider.ask/ask_with_retry. The run is still
    # labeled with (and compared by) the single model the user picked;
    # rotation is only an internal distribution mechanism so the workload
    # doesn't exhaust one shared free pool. Set False to always call
    # exactly the selected model (useful with a paid key / own credit).
    openrouter_model_rotation_enabled: bool = True

    # Task Readiness (#48/#58) -- Playwright-driven AI-agent harness. See
    # citepulse/task_readiness/ for how these are used; kept as plain
    # settings (not hardcoded constants) so a slower/CPU-only machine can
    # loosen timeouts without a code change.
    task_readiness_headless: bool = True
    task_readiness_max_tasks: int = 6
    task_readiness_max_task_runs: int = 6
    task_readiness_default_max_steps: int = 12
    task_readiness_page_action_timeout: float = 15.0
    task_readiness_navigation_timeout: float = 20.0
    task_readiness_ai_timeout: float = 60.0
    task_readiness_ai_max_retries: int = 2
    task_readiness_ai_retry_base_delay: float = 1.0
    task_readiness_call_delay_seconds: float = 0.0
    task_readiness_min_sample_size: int = 3
    task_readiness_user_agent: str = (
        "CitePulseBot/0.1 (+https://github.com/alsanjayllm/CitePulse)"
    )
    # FR-7 per-action capture: when True, the task harness captures a
    # best-effort viewport screenshot at every action (and records the
    # acted-on element selector) so a failed action's screenshot can support
    # failure-subtype diagnosis (see citepulse.failure_taxonomy). Off by
    # default because it adds a render per action -- bounded runtime under
    # NFR-1 unless explicitly enabled. `selector` capture is always on
    # (cheap -- no I/O); only the screenshot is gated here.
    task_readiness_step_screenshots: bool = False
    # FR-7 sampled task-stability measurement: when enabled, a small
    # (task_stability_sample_size) subset of a run's tasks is repeated
    # task_stability_repeats times so each sampled task gets a stability
    # label/success-rate (see harness.TaskRunResult.stability_label). Off by
    # default and deliberately a *sampled* subset (never every task re-run
    # 3x) so enabling it stays bounded per NFR-1.
    task_stability_enabled: bool = False
    task_stability_sample_size: int = 2
    task_stability_repeats: int = 3

    # Field-review methodology hardening (judge reliability): when enabled,
    # each single-word-classifier LLM call in citation_rate.py
    # (_classify_recommendation/_classify_message_accuracy) and
    # citation_correctness.check_citation_correctness()'s entailment call
    # asks the same question a second time with a light rephrase, recording
    # both classifications plus a `self_consistent` bool in the evidence
    # dict's raw_data -- an observability signal only. The *first* call's
    # result stays authoritative for scoring either way; this never becomes
    # a voting mechanism. Off by default: it doubles the LLM call count for
    # every gated classifier site, so it's opt-in rather than a silent cost
    # increase.
    classifier_self_consistency_enabled: bool = False

    # "v1 core" enterprise-LAN edition: hides Compare Models/OpenRouter/
    # batch/multi-model UI surfaces and defaults the KPI picker to the
    # 6-KPI core set (see citepulse/ui/app.py, components.py,
    # pages/run_audit.py) -- a build/branding flag, not a feature
    # implementation. Read via get_settings() at the specific UI call
    # sites that need to branch; no new function parameters, no second
    # PyInstaller spec/build target. Default False means every existing
    # pip-install/dev-checkout behavior (full nav, all pickers) is
    # byte-for-byte unchanged; a packaged core zip ships an .env/
    # environment with CORE_EDITION_MODE=true baked in instead.
    core_edition_mode: bool = False

    playwright_install_timeout: float = 300.0
    # Egress proxy for the Playwright/Chromium browser (homepage screenshot
    # and task harness). A corporate endpoint behind an explicit/internal
    # proxy can point CitePulse at it so headless Chromium honors corporate
    # egress instead of attempting a raw outbound that the network firewall
    # gates (the root cause of the one-time consent prompt many corporate
    # users see). Accepts a Playwright proxy server string, e.g.
    # `http://proxy:8080` or `socks5://proxy:1080`. None = leave it to
    # Chromium's own environment auto-detection (HTTP_PROXY/HTTPS_PROXY,
    # WPAD/PAC, the OS system proxy) -- CitePulse stays out of the way when
    # no explicit proxy is needed. Threaded into every playwright.chromium.
    # launch() call site (citepulse/screenshot.py, citepulse/task_readiness/
    # harness.py), never into a settings file / .env titled "secret".
    # Note: None is NOT the same as "force a direct connection" -- when this
    # is unset, Chromium/Playwright still honor inherited HTTP_PROXY/
    # HTTPS_PROXY/ALL_PROXY env vars, so a machine whose environment exports
    # a corporate proxy already routes the browser through it. To force a
    # direct connection you'd have to unset those env vars, not clear this
    # field. Proxy *authentication* (user/pass, NTLM) isn't supported here:
    # this threads only the `server` URL -- a proxy requiring credentials or
    # PAC/WPAD-based auth is out of scope and won't work via this field.
    egress_proxy: str | None = None

    @field_validator("egress_proxy")
    @classmethod
    def _validate_egress_proxy(cls, value: str | None) -> str | None:
        """Rejects a malformed egress_proxy (a scheme outside http(s)/socks5,
        or no host) at Settings-construction time -- a typo'd value should
        fail fast with a clear message rather than reaching Chromium's
        chromium.launch() and surfacing as an opaque launch error. Mirrors
        the codebase's "reject InvalidSiteURL eagerly" posture for URLs."""
        if value is None or not value.strip():
            return None
        parts = urlsplit(value)
        if (
            parts.scheme not in ("http", "https", "socks5", "socks5h")
            or not parts.hostname
        ):
            raise ValueError(
                f"egress_proxy={value!r} is not a valid proxy server URL -- "
                "use e.g. http://proxy:8080, https://proxy:443, or "
                "socks5://proxy:1080."
            )
        return value

    @property
    def data_dir(self) -> Path:
        path = Path(self.citepulse_data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'citepulse.db').as_posix()}"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def resolve_model(model: str | None) -> str:
    """The one place `model or settings.ollama_model` is decided --
    every Track C call site (audit.py, harness.py's run_task, runner.py's
    exception-path fallback) uses this instead of re-deriving the rule
    independently, so they can't drift out of sync."""
    return model or get_settings().ollama_model
