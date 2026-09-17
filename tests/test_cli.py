import subprocess

import pytest
import respx
from click.testing import CliRunner
from httpx import Response
from sqlmodel import Session, select

import citepulse.audit as audit_module
import citepulse.cli as cli_module
import citepulse.db as db_module
import citepulse.settings as settings_module
import citepulse.sites as sites_module
from citepulse.ai_engines import citation_rate as citation_rate_module
from citepulse.kpis import kpi_48, kpi_58
from citepulse.models import AuditRun, AuditRunModel
from citepulse.task_readiness.runner import TaskReadinessTrace


def _reset_singletons(monkeypatch, tmp_path):
    """settings.get_settings() and db.get_engine() are process-wide
    singletons -- force them to re-initialize against an isolated,
    per-test data dir instead of leaking state between tests."""
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


@pytest.fixture(autouse=True)
def _no_real_screenshot_capture(monkeypatch):
    """A `citepulse audit` CLI test has no business launching a real
    Playwright browser against the network -- same "no real browser
    anywhere in the test suite" posture as _stub_out_task_readiness below,
    just for the new screenshot-capture step instead of task readiness."""
    monkeypatch.setattr(audit_module, "capture_homepage_screenshot", lambda url: None)


def _stub_out_task_readiness(monkeypatch):
    """#48/#58 are exercised in depth in tests/test_kpi_48.py and
    tests/test_kpi_58.py against hand-built traces -- a full `citepulse
    audit` CLI test has no business launching a real (here: not even
    installed) Playwright browser, so their shared trace is stubbed to an
    instant unavailable result, same "no real browser anywhere in the
    test suite" posture as tests/test_task_readiness/."""
    trace = TaskReadinessTrace(
        available=False, unavailable_reason="stubbed for this test"
    )
    monkeypatch.setattr(
        kpi_48,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )
    monkeypatch.setattr(
        kpi_58,
        "gather_task_readiness_trace",
        lambda run_id, url, model=None, company_profile=None: trace,
    )


class _FakeStream:
    """A minimal stand-in carrying only what `_console_safe` reads
    (`.encoding`) -- deliberately not the real global `sys.stdout`/
    `sys.stderr`, so this test can't break anything else that happens to
    write to the console during the test run."""

    encoding = "cp1252"


def test_console_safe_replaces_characters_outside_console_encoding():
    """A real audit's Markdown report can contain characters (from
    crawled-site markup or an LLM's own output) outside Windows' default
    console codepage -- click.echo() would otherwise crash the whole
    `audit` command with UnicodeEncodeError after the run's real work is
    already saved. cp1252 can't encode U+21B5 ("↵"), the exact
    character that triggered this in the wild (see test_cli.py's sibling
    test below for the full scenario)."""
    result = cli_module._console_safe(
        "Log in↵        ↵      done", stream=_FakeStream()
    )

    assert "↵" not in result
    assert "Log in" in result
    assert "done" in result


def test_console_safe_defaults_to_stdout_when_no_stream_given():
    """The default (no `stream` passed) must read the real `sys.stdout`
    encoding, so the success-path `audit` command's `click.echo(...)`
    call site doesn't need to pass one explicitly."""
    assert cli_module._console_safe("hello") == "hello"


def test_console_safe_used_for_error_messages_too():
    """The fix's first cut only sanitized the success-path report; a
    UnicodeEncodeError in an error message (e.g. a non-cp1252 character
    surfaced from a wrapped exception) would still crash
    ClickException.show(), which prints to stderr -- the one path where
    the user most needs the message to actually appear."""
    result = cli_module._console_safe("Audit failed: bad ↵ url", stream=_FakeStream())

    assert "↵" not in result
    assert "Audit failed" in result


@respx.mock
def test_audit_command_completes_and_prints_report(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )
    # KPI #22's homepage fetch -- no search results configured, so its
    # evidence is unavailable (not exercised in depth here; see
    # tests/test_kpi_22.py for that KPI's own behavior). Also the fetch
    # citepulse.company_profile.extract_company_profile() makes for
    # Track B's CLI auto-accept path (this bare HTML has no title/
    # description/nav, so it resolves to the placeholder without an
    # extra Ollama call).
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><head></head></html>")
    )
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])
    # Track B's per-finding/executive narrative generation each make one
    # Ollama call on the success path -- mocked here (rather than left to
    # hit a real local Ollama) so this CLI-level test stays fast and
    # network-free; the wording itself is covered by
    # tests/test_business_narrative.py.
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This is worth prioritizing."}}
        )
    )

    result = CliRunner().invoke(cli_module.cli, ["audit", "https://example.com"])

    assert result.exit_code == 0
    assert "KPI #46" in result.output
    assert "Band: critical" in result.output
    assert "KPI #22" in result.output
    assert "KPI #48" in result.output
    assert "KPI #58" in result.output

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 1
        assert runs[0].status == "completed"


def test_audit_command_format_flag_dispatches_to_requested_renderer(
    monkeypatch, tmp_path
):
    """Phase 6: the `audit` command's new --format flag must route the
    already-gathered report data to the matching renderer, while the default
    (no --format) keeps using the Markdown renderer so existing scripts/CI
    are unchanged. run_audit and gather_report_data are stubbed so no
    network or real run is involved."""
    from types import SimpleNamespace

    _reset_singletons(monkeypatch, tmp_path)
    fake_run = SimpleNamespace(id="00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(
        cli_module, "run_audit_with_auto_review", lambda *a, **k: fake_run
    )
    monkeypatch.setattr(
        cli_module,
        "gather_report_data",
        lambda session, rid, detail="concise": {"run": fake_run},
    )

    calls = {}

    def _mk(name):
        def fn(data):
            calls[name] = data
            return f"<{name}>"

        return fn

    monkeypatch.setattr(cli_module, "render_json_report", _mk("json"))
    monkeypatch.setattr(cli_module, "render_csv_report", _mk("csv"))
    monkeypatch.setattr(cli_module, "render_html_report", _mk("html"))
    monkeypatch.setattr(cli_module, "render_markdown_report", _mk("markdown"))

    for fmt in ("json", "csv", "html"):
        calls.clear()
        result = CliRunner().invoke(
            cli_module.cli,
            ["audit", "https://example.com", "--format", fmt],
        )
        assert result.exit_code == 0, result.output
        assert f"<{fmt}>" in result.output
        assert calls.get(fmt) == {"run": fake_run}
        assert len(calls) == 1

    # Default (no --format) must stay on the Markdown renderer.
    calls.clear()
    result = CliRunner().invoke(cli_module.cli, ["audit", "https://example.com"])
    assert result.exit_code == 0, result.output
    assert "<markdown>" in result.output
    assert "markdown" in calls and len(calls) == 1


def test_audit_command_detail_flag_threads_into_gather_report_data(
    monkeypatch, tmp_path
):
    """The new `--detail concise|full` flag must thread straight into
    gather_report_data(session, run.id, detail=...) -- default 'concise'
    when omitted -- for every --format, including 'json'/'csv' (where the
    renderers simply ignore data['detail']/data['detailed'], so the flag
    is accepted but a no-op on their output)."""
    from types import SimpleNamespace

    _reset_singletons(monkeypatch, tmp_path)
    fake_run = SimpleNamespace(id="00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(
        cli_module, "run_audit_with_auto_review", lambda *a, **k: fake_run
    )

    captured = {}

    def _fake_gather(session, run_id, detail="concise"):
        captured["detail"] = detail
        return {"run": fake_run}

    monkeypatch.setattr(cli_module, "gather_report_data", _fake_gather)
    monkeypatch.setattr(cli_module, "render_markdown_report", lambda data: "<md>")
    monkeypatch.setattr(cli_module, "render_json_report", lambda data: "<json>")
    monkeypatch.setattr(cli_module, "render_csv_report", lambda data: "<csv>")

    # Default: no --detail -> "concise".
    result = CliRunner().invoke(cli_module.cli, ["audit", "https://example.com"])
    assert result.exit_code == 0, result.output
    assert captured["detail"] == "concise"

    # --detail full threads through for the Markdown/HTML path.
    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--detail", "full"]
    )
    assert result.exit_code == 0, result.output
    assert captured["detail"] == "full"

    # --format json/csv --detail full: the flag is accepted, and output
    # comes from the (stubbed) json/csv renderer unchanged -- the plan's
    # "no-op on output for json/csv" contract.
    for fmt, expected in (("json", "<json>"), ("csv", "<csv>")):
        result = CliRunner().invoke(
            cli_module.cli,
            ["audit", "https://example.com", "--format", fmt, "--detail", "full"],
        )
        assert result.exit_code == 0, result.output
        assert expected in result.output
        assert captured["detail"] == "full"

    # An invalid --detail value is a clean CLI error, not a traceback.
    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--detail", "bogus"]
    )
    assert result.exit_code != 0


def test_audit_command_auto_accepts_company_profile_review_on_first_run(
    monkeypatch, tmp_path
):
    """Track B's CLI-only behavior: no interactive prompt exists, so a
    brand-new site's company profile is extracted and auto-marked
    reviewed in the same `citepulse audit` invocation, rather than
    blocking (that's the Streamlit UI's job)."""
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    # This is the auto-accept path (sites.auto_resolve_context_review),
    # shared with the UI's batch-audit mode -- not cli.py's own
    # extract_company_profile import (that's only used by --refresh-profile).
    monkeypatch.setattr(
        sites_module, "extract_company_profile", lambda url: "Sells widgets online."
    )
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])

    with respx.mock:
        respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
        respx.get("https://example.com/.well-known/llms.txt").mock(
            return_value=Response(404)
        )
        respx.get("https://example.com").mock(
            return_value=Response(200, text="<html><head></head></html>")
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=Response(
                200, json={"message": {"content": "This is worth prioritizing."}}
            )
        )
        result = CliRunner().invoke(cli_module.cli, ["audit", "https://example.com"])

    assert result.exit_code == 0, result.output

    with Session(db_module.get_engine()) as session:
        from citepulse.sites import get_or_create_site

        site = get_or_create_site(session, "https://example.com")
        assert site.context_reviewed is True
        assert site.company_profile == "Sells widgets online."


def test_audit_command_marks_run_failed_not_stuck_running(monkeypatch, tmp_path):
    """The bug this test guards against: a KPI runner exception used to
    leave the AuditRun permanently at status="running" with a raw
    traceback shown to the user instead of a clean error."""
    _reset_singletons(monkeypatch, tmp_path)
    # This test isn't exercising crawler/Ollama behavior at all -- stub
    # out Track B's CLI auto-accept extraction step (which would
    # otherwise hit the real network) with a canned string.
    monkeypatch.setattr(
        cli_module, "extract_company_profile", lambda url: "A test company profile."
    )

    def _boom(audit_run_id, site_url):
        raise RuntimeError("simulated KPI crash")

    monkeypatch.setattr(audit_module, "_IMPLEMENTED_KPI_RUNNERS", [_boom])

    result = CliRunner().invoke(cli_module.cli, ["audit", "https://example.com"])

    assert result.exit_code != 0
    assert "Audit failed" in result.output

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun)).all()
        assert len(runs) == 1
        assert runs[0].status == "failed"


@respx.mock
def test_audit_command_model_flag_reaches_run_audit(monkeypatch, tmp_path):
    """Track C: `citepulse audit <url> --model <name>` must reach
    run_audit(..., model="custom-model") -- verified here by asserting the
    persisted AuditRun.model, an integration-style check against the real
    click command (not a mock of run_audit itself)."""
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><head></head></html>")
    )
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This is worth prioritizing."}}
        )
    )

    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--model", "custom-model"]
    )

    assert result.exit_code == 0, result.output

    with Session(db_module.get_engine()) as session:
        run = session.exec(select(AuditRun)).one()
        assert run.model == "custom-model"


@respx.mock
def test_audit_command_kpis_flag_restricts_to_requested_ids(monkeypatch, tmp_path):
    """`citepulse audit <url> --kpis 46,22` must reach
    run_audit(..., kpi_ids=[46, 22]) -- verified via the persisted
    manifest's requested_kpi_ids and the fact that only those KPIs'
    output appears."""
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><head></head></html>")
    )
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This is worth prioritizing."}}
        )
    )

    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--kpis", "46,22"]
    )

    assert result.exit_code == 0, result.output
    assert "KPI #46" in result.output
    assert "KPI #22" in result.output
    assert "KPI #24" not in result.output
    assert "KPI #48" not in result.output
    assert "KPI #58" not in result.output

    with Session(db_module.get_engine()) as session:
        run = session.exec(select(AuditRun)).one()
        assert run.manifest["requested_kpi_ids"] == [46, 22]


def test_audit_command_kpis_flag_with_unknown_id_is_a_clean_error(
    monkeypatch, tmp_path
):
    """An unknown --kpis id must surface as a clean click.ClickException
    (run_audit()'s own ValueError), not a raw traceback."""
    _reset_singletons(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module, "extract_company_profile", lambda url: "A test company profile."
    )

    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--kpis", "99"]
    )

    assert result.exit_code != 0
    assert "Unknown KPI id" in result.output


def test_audit_command_kpis_flag_with_non_numeric_value_is_a_clean_error(
    monkeypatch, tmp_path
):
    _reset_singletons(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--kpis", "abc"]
    )

    assert result.exit_code != 0
    assert "Invalid --kpis value" in result.output


def test_audit_command_refresh_profile_flag_forces_reextraction(monkeypatch, tmp_path):
    """--refresh-profile must re-extract company_profile even when a site
    is already reviewed with a stale value -- unlike the default
    auto-accept path (sites.auto_resolve_context_review), which only
    extracts when company_profile is still None."""
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    db_module.init_db()

    with Session(db_module.get_engine()) as session:
        from citepulse.sites import get_or_create_site

        site = get_or_create_site(session, "https://example.com")
        site.company_profile = "A stale profile."
        site.context_reviewed = True
        session.add(site)
        session.commit()

    extract_calls = []

    def _fake_extract(url):
        extract_calls.append(url)
        return "A freshly re-extracted profile."

    monkeypatch.setattr(cli_module, "extract_company_profile", _fake_extract)
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])

    with respx.mock:
        respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
        respx.get("https://example.com/.well-known/llms.txt").mock(
            return_value=Response(404)
        )
        respx.get("https://example.com").mock(
            return_value=Response(200, text="<html><head></head></html>")
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=Response(
                200, json={"message": {"content": "This is worth prioritizing."}}
            )
        )
        result = CliRunner().invoke(
            cli_module.cli, ["audit", "https://example.com", "--refresh-profile"]
        )

    assert result.exit_code == 0, result.output
    assert extract_calls == ["https://example.com"]

    with Session(db_module.get_engine()) as session:
        from citepulse.sites import get_or_create_site

        site = get_or_create_site(session, "https://example.com")
        assert site.company_profile == "A freshly re-extracted profile."
        assert site.context_reviewed is True


def test_sites_list_and_remove(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    empty = CliRunner().invoke(cli_module.cli, ["sites", "list"])
    assert empty.exit_code == 0
    assert "No sites tracked yet" in empty.output

    with Session(db_module.get_engine()) as session:
        from citepulse.sites import get_or_create_site

        get_or_create_site(session, "https://example.com")

    listed = CliRunner().invoke(cli_module.cli, ["sites", "list"])
    assert listed.exit_code == 0
    assert "https://example.com" in listed.output

    removed = CliRunner().invoke(
        cli_module.cli, ["sites", "remove", "https://example.com"]
    )
    assert removed.exit_code == 0
    assert "Removed" in removed.output

    after = CliRunner().invoke(cli_module.cli, ["sites", "list"])
    assert "No sites tracked yet" in after.output


def test_sites_remove_unknown_url_is_a_clean_error(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli_module.cli, ["sites", "remove", "https://not-tracked.com"]
    )

    assert result.exit_code != 0
    assert "No tracked site matches" in result.output


def test_ui_command_without_streamlit_gives_a_clean_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("no module named streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = CliRunner().invoke(cli_module.cli, ["ui"])

    assert result.exit_code != 0
    assert "citepulse[ui]" in result.output


def test_ui_command_forwards_log_level_to_the_streamlit_subprocess(monkeypatch):
    """`citepulse ui` launches Streamlit as a separate process
    (subprocess.run) that never inherits this process's own
    configure_logging() call -- a --log-level passed at the group level
    must cross via an env var instead, or it silently does nothing for
    the actual UI process."""
    pytest.importorskip("streamlit")
    captured = {}

    def _fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(cli_module.subprocess, "run", _fake_run)

    result = CliRunner().invoke(cli_module.cli, ["--log-level", "DEBUG", "ui"])

    assert result.exit_code == 0, result.output
    assert captured["env"]["LOG_LEVEL"] == "DEBUG"


def test_ui_command_without_log_level_flag_does_not_force_an_env_override(monkeypatch):
    pytest.importorskip("streamlit")
    captured = {}

    def _fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(cli_module.subprocess, "run", _fake_run)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    result = CliRunner().invoke(cli_module.cli, ["ui"])

    assert result.exit_code == 0, result.output
    assert "LOG_LEVEL" not in captured["env"]


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def test_setup_installs_db_and_playwright_chromium_successfully(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0),
    )

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "Chromium installed" in result.output


def test_setup_playwright_frozen_build_uses_inprocess_installer(monkeypatch, tmp_path):
    """In a PyInstaller build, sys.executable IS citepulse.exe itself --
    a `-m playwright install` subprocess would run the packaged exe with
    bogus arguments instead of installing anything. Since the "v1 core"
    packaging feature, the frozen branch no longer just skips with a
    pip-install-instead message -- it calls
    playwright.__main__.main() in-process on a background thread instead
    (confirmed against a real frozen-build spike during that feature's
    own verification), never shelling out via subprocess.run."""
    _reset_singletons(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module.sys, "frozen", True, raising=False)
    subprocess_calls = []
    monkeypatch.setattr(
        cli_module.subprocess, "run", lambda *a, **k: subprocess_calls.append(1)
    )

    import playwright.__main__ as pw_main_module

    monkeypatch.setattr(
        pw_main_module, "main", lambda: (_ for _ in ()).throw(SystemExit(0))
    )

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "Chromium installed" in result.output
    assert subprocess_calls == []  # never shelled out to `-m playwright`


def test_setup_playwright_frozen_build_install_failure_does_not_raise(
    monkeypatch, tmp_path
):
    """A nonzero exit / real exception from the in-process installer must
    still satisfy _ensure_playwright_chromium's "never raises, always
    returns bool" contract -- `setup` itself must not fail."""
    _reset_singletons(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_module.sys, "frozen", True, raising=False)

    import playwright.__main__ as pw_main_module

    monkeypatch.setattr(
        pw_main_module, "main", lambda: (_ for _ in ()).throw(SystemExit(1))
    )

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "install failed" in result.output


def test_setup_playwright_missing_module_does_not_fail_setup(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    def _raise_not_found(*a, **k):
        raise FileNotFoundError("no playwright")

    monkeypatch.setattr(cli_module.subprocess, "run", _raise_not_found)

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "reinstall CitePulse" in result.output


def test_setup_playwright_timeout_does_not_fail_setup(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)

    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["playwright"], timeout=1)

    monkeypatch.setattr(cli_module.subprocess, "run", _raise_timeout)

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "timed out" in result.output


def test_setup_playwright_nonzero_exit_does_not_fail_setup(monkeypatch, tmp_path):
    _reset_singletons(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="disk full"),
    )

    result = CliRunner().invoke(cli_module.cli, ["setup"])

    assert result.exit_code == 0
    assert "Database initialized" in result.output
    assert "install failed" in result.output
    assert "disk full" in result.output


@respx.mock
def test_audit_command_models_flag_creates_multi_model_runs(monkeypatch, tmp_path):
    """FR-3: `citepulse audit <url> --models a,b,c` must run the body once
    per model on the CLI, persisting one primary AuditRun plus two child
    runs (parent_run_id links) with AuditRunModel rows -- verified via the
    DB, against the real click command (not a mock of run_audit)."""
    _reset_singletons(monkeypatch, tmp_path)
    _stub_out_task_readiness(monkeypatch)
    respx.get("https://example.com/llms.txt").mock(return_value=Response(404))
    respx.get("https://example.com/.well-known/llms.txt").mock(
        return_value=Response(404)
    )
    respx.get("https://example.com").mock(
        return_value=Response(200, text="<html><head></head></html>")
    )
    monkeypatch.setattr(citation_rate_module, "search", lambda query, **kwargs: [])
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=Response(
            200, json={"message": {"content": "This is worth prioritizing."}}
        )
    )

    result = CliRunner().invoke(
        cli_module.cli, ["audit", "https://example.com", "--models", "a,b,c"]
    )

    assert result.exit_code == 0, result.output

    with Session(db_module.get_engine()) as session:
        runs = session.exec(select(AuditRun).order_by(AuditRun.model)).all()
        assert [r.model for r in runs] == ["a", "b", "c"]
        primary = next(r for r in runs if r.model == "a")
        assert primary.parent_run_id is None
        assert all(r.parent_run_id == primary.id for r in runs if r.model in ("b", "c"))
        arm_rows = session.exec(
            select(AuditRunModel).order_by(AuditRunModel.sequence)
        ).all()
        assert len(arm_rows) == 3
        assert [arm.role for arm in arm_rows] == ["primary", "compared", "compared"]


def test_prompts_import_then_validate(monkeypatch, tmp_path):
    """Phase 3 (FR-2): `citepulse prompts import` ingests a prompt library
    file into the site's active PromptItem corpus (versioned, previous set
    deactivated) and `citepulse prompts validate` runs the quality report
    against it -- against the real click commands and the real DB."""
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    import json
    from citepulse.models import PromptItem

    defs = [
        {
            "text": "What should a {buyer} look for?",
            "intent": "awareness",
            "topic_cluster": "product",
        },
        {
            "text": "Which rival is cheapest and?",
            "intent": "evaluation",
            "topic_cluster": "pricing",
        },
    ]
    pfile = tmp_path / "prompts.json"
    pfile.write_text(json.dumps({"prompts": defs}), encoding="utf-8")

    imported = CliRunner().invoke(
        cli_module.cli, ["prompts", "import", "https://example.com", str(pfile)]
    )
    # 2 prompts and one placeholder block one truncation flag -> invalid.
    assert imported.exit_code != 0
    assert "imported 2 prompts" in imported.output
    assert "placeholder/template artefact" in imported.output

    with Session(db_module.get_engine()) as session:
        rows = session.exec(select(PromptItem)).all()
        assert len(rows) == 2
        assert all(r.active is True for r in rows)
        assert all(r.version == 1 for r in rows)
        assert any("{buyer}" in r.text for r in rows)

        # validate the active set through the same DB
        from citepulse.prompts import validate_site_prompts

        report = validate_site_prompts(session, rows[0].site_id)
        assert report.count == 2
        assert report.valid is False

    validated = CliRunner().invoke(
        cli_module.cli, ["prompts", "validate", "https://example.com"]
    )
    assert validated.exit_code != 0  # advisory but reports the failing set
    assert "prompt set: 2 prompts" in validated.output
