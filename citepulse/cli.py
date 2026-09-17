import os
import subprocess
import sys

import click
from sqlmodel import select

from citepulse.audit import MissingOpenRouterKey
from citepulse.company_profile import extract_company_profile
from citepulse.competitor_discovery import (
    commit_competitor_candidates,
    discover_competitors,
)
from citepulse.competitors import add_competitor, list_competitors, remove_competitor
from citepulse.db import get_session, init_db
from citepulse.logging_setup import configure_logging
from citepulse.models import Site
from citepulse.reporting import (
    build_kpi_trend,
    gather_report_data,
    render_csv_report,
    render_html_report,
    render_json_report,
    render_markdown_report,
)
from citepulse.settings import get_settings
from citepulse.sites import (
    InvalidSiteURL,
    SiteLimitExceeded,
    get_or_create_site,
    list_sites,
    normalize_url,
    remove_site,
    run_audit_with_auto_review,
)


@click.group()
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help=(
        "Override the LOG_LEVEL setting for this invocation. Logs are "
        "written to a local file under the CitePulse data dir -- never "
        "sent anywhere."
    ),
)
@click.pass_context
def cli(ctx: click.Context, log_level: str | None):
    """CitePulse — local-first, open-source AEO audit tool."""
    configure_logging(level=log_level)
    # Stashed for ui() below: that command launches Streamlit in a
    # separate process (subprocess.run), which never inherits this
    # process's configure_logging() call -- an explicit --log-level here
    # has to be forwarded to that child process itself, not just applied
    # to this one.
    ctx.obj = log_level


def _console_safe(text: str, stream=None) -> str:
    """A real audit's Markdown report -- or an error message wrapping a
    crawled-site/LLM-derived exception -- can contain arbitrary Unicode,
    since crawled-site markup or an LLM's own output can both introduce
    characters outside Windows' default console codepage (cp1252).
    click.echo()/ClickException.show() write straight through to the
    console and raise UnicodeEncodeError on the first unencodable
    character, crashing the CLI (on the success path, after the run's
    real work is already saved). `stream` defaults to stdout (the
    success-path report); pass `sys.stderr` for error messages, since
    that's what Click actually prints them to. Replacing rather than
    crashing keeps this a display-only limitation."""
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _ensure_playwright_chromium_frozen(echo=click.echo) -> bool:
    """Frozen-build branch of _ensure_playwright_chromium(): there's no
    real Python interpreter to spawn a `python -m playwright install
    chromium` subprocess against (sys.executable IS citepulse.exe in a
    PyInstaller build), so this calls Playwright's own installer
    in-process instead (`playwright.__main__.main(["install",
    "chromium"])`) -- confirmed working against a throwaway PyInstaller
    spike (see docs/PACKAGING.md) including the actual browser download
    and a real chromium.launch() from inside a frozen onedir tree.

    PLAYWRIGHT_BROWSERS_PATH=0 is set (if not already set) before that
    call so the frozen driver installs browsers into a
    `.local-browsers` directory next to itself rather than
    %LOCALAPPDATA%\\ms-playwright -- the spike showed a frozen driver's
    default resolution differs from a normal pip install's, and this
    keeps the whole onedir folder self-contained (matching "unzip and
    run" -- no dependency on a user-profile cache directory living
    outside the folder the user was told to keep together).

    Runs on a background thread with a join timeout (rather than
    subprocess.run's timeout=, since there is no child process to
    terminate) -- if the timeout elapses the installer thread is simply
    abandoned (daemon=True) and this returns False; the download may
    still finish in the background, but the caller is never blocked past
    the configured timeout. Same never-raises/always-returns-bool/
    always-echoes contract as the non-frozen path.

    Deliberately does NOT redirect sys.stdout/sys.stderr around the
    installer call (an earlier draft did, to capture output for the
    failure message below, and a /code-review pass on this very function
    caught why that's wrong): those are process-global, not thread-local,
    so a redirect made on this background thread is visible everywhere,
    including the main thread's own console output -- and if the
    download outlives the join timeout, the abandoned thread's `with`
    block stays open indefinitely with no bound on when it un-redirects
    them again, racing with whatever `_launch_streamlit()` does
    immediately afterward in the main thread. Confirmed via a real frozen
    build that this isn't needed anyway: `playwright.__main__.main()`'s
    own download progress bars come from a child process's inherited file
    descriptors, not Python's sys.stdout object, so they reach the real
    console either way; only a genuine Python-level exception's own text
    is lost by not capturing it, and str(exc) below covers that."""
    import threading

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")
    echo("Installing Playwright's Chromium browser (first run may take a minute)...")

    result: dict = {}

    def _run_install() -> None:
        from playwright.__main__ import main as pw_main

        # playwright.__main__.main() takes no arguments -- it reads
        # sys.argv itself (mirroring the real `python -m playwright ...`
        # CLI entry point), confirmed against the throwaway spike above.
        # sys.argv is a process-global mutated from this background
        # thread -- safe in practice (not in principle) because
        # pw_main() reads it synchronously into a subprocess.run() args
        # list *before* that call blocks for the actual download, so by
        # the time this thread could still be alive past the join
        # timeout below (i.e. during the download itself), sys.argv has
        # already been consumed and _launch_streamlit()'s own later
        # `sys.argv = [...]` reassignment in the main thread can't affect
        # an already-launched subprocess's argument list.
        sys.argv = ["playwright", "install", "chromium"]
        try:
            pw_main()
        except SystemExit as exc:
            result["exit_code"] = exc.code if isinstance(exc.code, int) else 0
        except Exception as exc:  # pragma: no cover -- defensive, never raise
            result["exception"] = exc
        else:
            result["exit_code"] = 0

    thread = threading.Thread(target=_run_install, daemon=True)
    thread.start()
    thread.join(timeout=get_settings().playwright_install_timeout)

    if thread.is_alive():
        echo(
            "Playwright's Chromium download timed out -- check your network "
            "connection and try again from the app, or run "
            '"Start CitePulse.bat" again later.'
        )
        return False

    if "exception" in result:
        echo(
            "Playwright's Chromium install failed -- Task Readiness KPIs "
            f"(#48/#58) will be unavailable until it succeeds. Details: "
            f"{result['exception']}"
        )
        return False

    if result.get("exit_code", 1) != 0:
        echo(
            "Playwright's Chromium install failed (exit code "
            f"{result.get('exit_code')}) -- Task Readiness KPIs (#48/#58) "
            "will be unavailable until it succeeds. See the console output "
            "above for details."
        )
        return False

    echo("Chromium installed.")
    return True


def _ensure_playwright_chromium(echo=click.echo) -> bool:
    """Runs `python -m playwright install chromium` (idempotent -- a
    no-op if already installed) so KPI #48/#58's Task Readiness harness
    has a browser to drive. Never raises and never fails `setup`/`launch`
    itself on any failure mode -- a missing browser only blocks #48/#58,
    not the rest of the tool -- it just echoes a clear instruction and
    returns False. Frozen builds branch to an in-process installer
    (_ensure_playwright_chromium_frozen) instead of a subprocess -- see
    that function's own docstring."""
    if getattr(sys, "frozen", False):
        return _ensure_playwright_chromium_frozen(echo)

    echo("Installing Playwright's Chromium browser (first run may take a minute)...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=get_settings().playwright_install_timeout,
        )
    except FileNotFoundError:
        echo(
            "Could not run Playwright's installer (module not found) -- "
            'reinstall CitePulse\'s dependencies with `pip install -e ".[dev]"` '
            "and re-run `citepulse setup`, or install the browser manually with "
            "`python -m playwright install chromium`."
        )
        return False
    except subprocess.TimeoutExpired:
        echo(
            "Playwright's Chromium download timed out -- check your network "
            "connection and re-run `citepulse setup`, or install manually with "
            "`python -m playwright install chromium`."
        )
        return False

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-500:]
        echo(
            "Playwright's Chromium install failed -- Task Readiness KPIs "
            "(#48/#58) will be unavailable until it succeeds. Try running "
            "`python -m playwright install chromium` manually. "
            f"Details: {stderr_tail}"
        )
        return False

    echo("Chromium installed.")
    return True


@cli.command()
def setup():
    """Initialize the local database and install Playwright's Chromium
    browser (needed for the Task Readiness KPIs, #48/#58). Ollama
    auto-install and hardware preflight checks are not yet implemented."""
    init_db()
    click.echo("Database initialized. Run `citepulse audit <url>` to start.")
    _ensure_playwright_chromium()


def _refresh_company_profile_and_auto_accept(session, url) -> None:
    """--refresh-profile: unlike sites.auto_resolve_context_review (which
    only extracts when company_profile is still None), this always
    re-extracts -- the whole point of the flag is to overwrite a stale
    value -- then auto-accepts it immediately, same CLI posture as a
    first-run audit (no interactive prompt exists on this path)."""
    site = get_or_create_site(session, url)
    site.company_profile = extract_company_profile(site.url)
    site.context_reviewed = True
    session.add(site)
    session.commit()


@cli.command()
@click.argument("url")
@click.option(
    "--model",
    default=None,
    help=(
        "Ollama model to use for this run (default: the configured "
        "ollama_model setting)."
    ),
)
@click.option(
    "--models",
    default=None,
    help=(
        "Comma-separated models to run a multi-model (FR-3) audit, e.g. "
        "--models gpt-4o,claude-3.5,llama3. Overrides --model; produces one "
        "primary run plus a child run per model, consolidatable via the "
        "comparison report."
    ),
)
@click.option(
    "--refresh-profile",
    is_flag=True,
    default=False,
    help=(
        "Force a fresh company-profile extraction from the live homepage "
        "before this run, overwriting the stored value and auto-accepting "
        "it (same auto-accept posture as a first-run CLI audit -- no "
        "interactive review exists on the CLI)."
    ),
)
@click.option(
    "--kpis",
    default=None,
    help="Comma-separated KPI ids to run (default: all), e.g. --kpis 46,22,24",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json", "csv", "html"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help=(
        "Report output format. Default 'markdown' prints to stdout exactly as "
        "before, so existing scripts/CI keep working; 'json'/'csv' emit "
        "machine-readable data; 'html' prints the infographic report."
    ),
)
@click.option(
    "--detail",
    "detail",
    type=click.Choice(["concise", "full"], case_sensitive=False),
    default="concise",
    show_default=True,
    help=(
        "Report detail level. 'full' additionally surfaces internal "
        "processing evidence already gathered per KPI (per-probe LLM "
        "Q&A, per-citation fetch/entailment detail, per-path fetch "
        "diagnostics, per-task agent step traces) as a new 'Full Detail' "
        "section, capped at 5 items per family. Only affects "
        "'markdown'/'html' --format output; ignored for 'json'/'csv'."
    ),
)
def audit(
    url: str,
    model,
    models,
    refresh_profile: bool,
    kpis: str | None,
    output_format: str,
    detail: str,
):
    """Run an audit against URL and print a Markdown report."""
    init_db()

    kpi_ids = None
    if kpis:
        try:
            kpi_ids = [int(x.strip()) for x in kpis.split(",") if x.strip()]
        except ValueError as exc:
            raise click.ClickException(f"Invalid --kpis value: {kpis}") from exc

    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]

    with get_session() as session:
        try:
            if refresh_profile:
                _refresh_company_profile_and_auto_accept(session, url)
            run = run_audit_with_auto_review(
                session, url, model=model, models=model_list, kpi_ids=kpi_ids
            )
        except (SiteLimitExceeded, InvalidSiteURL) as exc:
            raise click.ClickException(
                _console_safe(str(exc), stream=sys.stderr)
            ) from exc
        except MissingOpenRouterKey as exc:
            # The CLI has no --api-key option today, so any openrouter:
            # --model is guaranteed to hit this -- a clear, actionable
            # message instead of the generic "Audit failed" wrapper below.
            raise click.ClickException(
                _console_safe(
                    f"{exc} The CLI does not currently support supplying "
                    "an OpenRouter API key -- use an Ollama model, or run "
                    "this audit from the Streamlit UI (`citepulse ui`) "
                    "instead.",
                    stream=sys.stderr,
                )
            ) from exc
        except ValueError as exc:
            # run_audit()'s own validation of an unknown-but-numeric
            # --kpis id (e.g. --kpis 99) -- a clean ClickException, not a
            # raw traceback, same posture as the SiteLimitExceeded/
            # InvalidSiteURL branch above.
            raise click.ClickException(
                _console_safe(str(exc), stream=sys.stderr)
            ) from exc
        except Exception as exc:
            raise click.ClickException(
                _console_safe(f"Audit failed: {exc}", stream=sys.stderr)
            ) from exc

        data = gather_report_data(session, run.id, detail=detail)
        if output_format == "json":
            click.echo(_console_safe(render_json_report(data)))
        elif output_format == "csv":
            click.echo(_console_safe(render_csv_report(data)))
        elif output_format == "html":
            click.echo(_console_safe(render_html_report(data)))
        else:
            click.echo(_console_safe(render_markdown_report(data)))


@cli.group()
def sites():
    """Manage the ≤10 sites tracked by CitePulse."""


@sites.command("list")
def sites_list():
    """List tracked sites."""
    init_db()
    with get_session() as session:
        rows = list_sites(session)
        if not rows:
            click.echo("No sites tracked yet. Run `citepulse audit <url>` to add one.")
            return
        for site in rows:
            click.echo(f"{site.url}  (added {site.created_at:%Y-%m-%d})")


@sites.command("remove")
@click.argument("url")
def sites_remove(url: str):
    """Remove a tracked site (and its audit history) to free a slot."""
    init_db()
    with get_session() as session:
        try:
            remove_site(session, url)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Removed {url}.")


@sites.command("trend")
@click.argument("url")
def sites_trend(url: str):
    """Show each KPI's value-over-time trend across this site's completed
    audit runs. Needs at least 2 completed runs -- prints a plain
    "not enough data yet" message rather than an error otherwise."""
    init_db()
    with get_session() as session:
        normalized = normalize_url(url)
        site = session.exec(select(Site).where(Site.url == normalized)).first()
        if site is None:
            raise click.ClickException(f"No tracked site matches {url!r}.")

        trend = build_kpi_trend(session, site.id)
        if trend is None:
            click.echo(
                f"Not enough completed audit runs for {url} yet to build a "
                "trend (need at least 2). Run `citepulse audit` again once "
                "more data is available."
            )
            return

        click.echo(f"Trend across {trend['run_count']} completed run(s) for {url}:")
        for entry in trend["kpis"]:
            points = " -> ".join(
                f"{p['started_at'].date()}: {p['value']}" for p in entry["points"]
            )
            click.echo(f"  KPI #{entry['kpi_id']} - {entry['kpi_name']}: {points}")


@cli.group()
def competitor():
    """Manage curated competitors tracked against a site (FR-1)."""


@competitor.command("add")
@click.argument("site_url")
@click.argument("competitor_url")
@click.option("--name", default=None, help="Optional human-readable label.")
def competitor_add(site_url: str, competitor_url: str, name: str | None):
    """Track `competitor_url` against a site (creating it if needed). Its
    canonical domain is fed into the citation/mention checks on every
    audit."""
    init_db()
    with get_session() as session:
        try:
            competitor_row = add_competitor(
                session, site_url, competitor_url, name=name
            )
        except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"Tracked {competitor_row.name} "
            f"({', '.join(competitor_row.canonical_domains)}) "
            f"against {site_url}."
        )


@competitor.command("list")
@click.argument("site_url")
def competitor_list(site_url: str):
    """List competitors tracked against a site."""
    init_db()
    with get_session() as session:
        rows = list_competitors(session, site_url)
        if not rows:
            click.echo(
                f"No competitors tracked against {site_url}. "
                "Run `citepulse competitor add <site_url> <competitor_url>`."
            )
            return
        for row in rows:
            status = "active" if row.active else "inactive"
            click.echo(
                f"{row.name}  ({', '.join(row.canonical_domains)})  "
                f"[{status}]  (added {row.created_at:%Y-%m-%d})"
            )


@competitor.command("remove")
@click.argument("site_url")
@click.argument("competitor_url")
def competitor_remove(site_url: str, competitor_url: str):
    """Stop tracking `competitor_url` against a site."""
    init_db()
    with get_session() as session:
        try:
            remove_competitor(session, site_url, competitor_url)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Removed {competitor_url} as a competitor of {site_url}.")


@competitor.command("discover")
@click.argument("site_url")
@click.option(
    "--max-candidates",
    default=8,
    show_default=True,
    type=int,
    help="Maximum number of proposed candidates.",
)
@click.option(
    "--add",
    "add_flag",
    is_flag=True,
    default=False,
    help=(
        "Also track every newly proposed (not-already-tracked) candidate. "
        "Without this flag, discovery is a dry run -- nothing is written."
    ),
)
def competitor_discover(site_url: str, max_candidates: int, add_flag: bool):
    """Propose competitors for `site_url` via 2 web searches + an LLM
    call, printing each candidate (dry run by default). Pass --add to
    also track every newly proposed candidate -- never a candidate
    already tracked, and never anything until you explicitly opt in."""
    init_db()
    with get_session() as session:
        try:
            candidates = discover_competitors(
                session, site_url, max_candidates=max_candidates
            )
        except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
            raise click.ClickException(str(exc)) from exc

        if not candidates:
            click.echo(
                f"No competitor candidates found for {site_url}. Try "
                "`citepulse competitor add <site_url> <competitor_url>` "
                "to track one manually."
            )
            return

        for candidate in candidates:
            tracked_tag = " [already tracked]" if candidate.already_tracked else ""
            click.echo(
                f"{candidate.name}  ({candidate.domain})  "
                f"confidence={candidate.confidence}{tracked_tag}\n"
                f"    {candidate.rationale}"
            )

        if not add_flag:
            click.echo(
                "\nDry run -- nothing was tracked. Re-run with --add to "
                "track the new candidates above."
            )
            return

        accept_domains = [c.domain for c in candidates if not c.already_tracked]
        try:
            added = commit_competitor_candidates(
                session, site_url, candidates, accept_domains
            )
        except (ValueError, InvalidSiteURL, SiteLimitExceeded) as exc:
            raise click.ClickException(str(exc)) from exc

        if added:
            click.echo(
                "\nTracked: "
                + ", ".join(f"{c.name} ({c.canonical_domains[0]})" for c in added)
            )
        else:
            click.echo("\nNothing new to track (all candidates already tracked).")


def _launch_streamlit(ctx: click.Context) -> None:
    """Shared body of `ui()` and `launch()`: resolves citepulse/ui/app.py
    and starts Streamlit against it, forwarding a --log-level override the
    same way in both frozen and non-frozen builds.

    Non-frozen: unchanged from before this feature -- a real `python -m
    streamlit run <app_path>` subprocess, env-forwarded LOG_LEVEL since a
    subprocess never inherits this process's own configure_logging() call.

    Frozen: no real Python interpreter to spawn a subprocess against
    (same reason _ensure_playwright_chromium_frozen exists), so this
    calls `streamlit.web.cli.main()` in-process instead, confirmed
    against a throwaway PyInstaller spike (see docs/PACKAGING.md) to
    actually serve a page once two sharp edges are handled:

    1. Streamlit's dev-mode auto-detection misfires when frozen
       (`RuntimeError: server.port does not work when
       global.developmentMode is true`) -- fixed by explicitly setting
       STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false before calling
       stcli.main(), confirmed by the spike to resolve it cleanly.
    2. stcli.main() calls sys.exit() internally on completion --
       SystemExit is caught here and re-raised as a ClickException on a
       non-zero code so a failure (e.g. port already in use) still
       surfaces as a clear CLI error instead of an unhandled traceback or
       a swallowed exit code; sys.argv is set explicitly first since
       stcli.main() reads it directly rather than accepting an argument
       list.

    The env-forwarding LOG_LEVEL hack from the non-frozen path is not
    needed in the frozen branch (re-verified via the spike): there is no
    subprocess boundary, so setting os.environ directly in this same
    process already reaches citepulse/ui/app.py's own configure_logging()
    call the normal way.

    Both branches pass --server.address=127.0.0.1 explicitly (a real
    packaged-build test surfaced Streamlit's own default of binding
    0.0.0.0 -- all interfaces -- which triggers a Windows Defender
    Firewall prompt on first launch and, if allowed, exposes the UI to
    the whole LAN with no authentication). CitePulse is a single-user
    local tool; nothing in its design calls for LAN-shared access, so
    this is forced rather than left to .streamlit/config.toml, which
    isn't guaranteed to be discovered from the frozen exe's working
    directory.
    """
    from pathlib import Path

    # Deliberately Path(citepulse.__file__), not Path(__file__): a
    # PyInstaller entry-point script is unpacked flat (its own __file__
    # resolves to something like _internal\cli.py, with no citepulse/
    # package prefix at all -- confirmed by a real frozen-build test
    # during this feature's own verification), while the `citepulse`
    # package itself still resolves correctly under _internal\citepulse\
    # since it's a real bundled package, not the entry script. Works
    # identically non-frozen (citepulse.__file__ is just
    # .../citepulse/__init__.py there too).
    import citepulse

    app_path = Path(citepulse.__file__).parent / "ui" / "app.py"

    if getattr(sys, "frozen", False):
        os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")
        if ctx.obj:
            os.environ["LOG_LEVEL"] = ctx.obj

        import streamlit.web.cli as stcli

        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.address=127.0.0.1",
        ]
        try:
            exit_code = stcli.main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 0
        if exit_code:
            raise click.ClickException(f"Streamlit exited with code {exit_code}.")
        return

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            'Streamlit UI not installed -- run `pip install "citepulse[ui]"`.'
        ) from exc

    # A --log-level passed to `citepulse ui`/`launch` itself lives in
    # ctx.obj (set by cli() above) -- this subprocess is a separate
    # process that never inherits that in-process configure_logging()
    # call, so the override has to cross via an env var instead.
    # citepulse/ui/app.py's own configure_logging() call already falls
    # back to settings.log_level (itself LOG_LEVEL-driven) when this
    # isn't set.
    env = os.environ.copy()
    if ctx.obj:
        env["LOG_LEVEL"] = ctx.obj
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address=127.0.0.1",
        ],
        env=env,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Streamlit exited with code {result.returncode}.")


@cli.command()
@click.pass_context
def ui(ctx: click.Context):
    """Launch the Streamlit UI (opens in your browser)."""
    _launch_streamlit(ctx)


@cli.command()
@click.pass_context
def launch(ctx: click.Context):
    """The one command a packaged zip's launcher ("Start CitePulse.bat")
    calls -- a non-technical end user never runs `setup`/`ui` separately
    or sees a CLI. Initializes the database, best-effort installs
    Playwright's Chromium browser (never blocks the launch on failure --
    Task Readiness KPIs #48/#58 just stay unavailable until it succeeds),
    then launches the Streamlit UI exactly like `citepulse ui`."""
    init_db()
    click.echo("Database initialized.")
    _ensure_playwright_chromium()
    _launch_streamlit(ctx)


@click.group()
def prompts():
    """Manage a site's custom prompt set (SRS FR-2 prompt quality control)."""


@prompts.command("validate")
@click.argument("site_url")
def prompts_validate(site_url: str):
    """Validate the active prompt set for SITE_URL against SRS FR-2/FR-2.5
    (structural + statistical checks). Exits non-zero when the set does not
    meet the tier floor."""
    from citepulse.prompts import validate_site_prompts

    with get_session() as session:
        site = get_or_create_site(session, site_url)
        report = validate_site_prompts(session, site.id)

    for error in report.errors:
        click.echo(f"  error: {error}")
    for warning in report.warnings:
        click.echo(f"  warning: {warning}")
    click.echo(
        f"prompt set: {report.count} prompts | tier {report.tier_required}+ | "
        f"per-cluster {report.min_per_cluster}+ | valid={report.valid}"
    )
    if not report.valid:
        raise click.ClickException("prompt set failed validation")


@prompts.command("import")
@click.argument("site_url")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def prompts_import(site_url: str, file: str):
    """Import and activate a prompt set for SITE_URL from FILE (.json or
    .csv), versioning it and deactivating the previous set. Validates the
    new set and prints the report; exits non-zero if it fails validation."""
    from citepulse.prompts import import_prompt_set, parse_prompt_file

    items = parse_prompt_file(file)
    if not items:
        raise click.ClickException("no prompt definitions found in file")

    with get_session() as session:
        site = get_or_create_site(session, site_url)
        report = import_prompt_set(session, site.id, items)

    for error in report.errors:
        click.echo(f"  error: {error}")
    for warning in report.warnings:
        click.echo(f"  warning: {warning}")
    click.echo(
        f"imported {report.count} prompts (from {len(items)} defs) as new version | "
        f"valid={report.valid}"
    )
    if not report.valid:
        raise click.ClickException(
            "prompt set imported but failed validation -- see errors above"
        )


cli.add_command(prompts)


if __name__ == "__main__":
    cli()
