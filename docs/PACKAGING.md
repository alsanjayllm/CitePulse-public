# Packaging: standalone Windows build

Why this exists: a corporate end user with no admin rights, and possibly
no Python installed, can't use `pip install -e ".[dev]"`. `build_dist.bat`
produces a self-contained Windows folder (bundled interpreter and all)
that such a user can unzip anywhere and run directly — no installer, no
PATH changes, no pip, no system Python.

## Build

```bash
build_dist.bat
```

Same conventions as `run_test.bat`: creates/reuses `.venv`, installs the
`build` extra (`pyinstaller`, kept out of `[dev]` so the normal test loop
doesn't pull PyInstaller's bootloader binaries), cleans `build/`/`dist/`,
runs PyInstaller against `citepulse.spec`, then zips the result to
`dist\citepulse-win-x64-<version>.zip` (version read from
`citepulse.__version__`, never hardcoded).

Run `run_test.bat` first to confirm tests pass — `build_dist.bat` doesn't
re-run pytest itself.

## Why onedir, not onefile

`citepulse.spec` builds a **onedir** output (a folder of the exe plus its
bundled DLLs/data), not a single self-extracting onefile exe. Onefile
unpacks itself to a fresh `%TEMP%` directory on *every* launch, which is
slower on every invocation and looks more like dropper behavior to AV/EDR
heuristics — exactly the wrong shape for a locked-down corporate machine.
The tradeoff: users must keep the whole unzipped folder together, not just
copy `citepulse.exe` on its own.

## Why `citepulse.spec` exists (not just CLI flags)

The two data-file entries and the hidden-imports list are exactly what a
`.spec` file is for — explicit, diffable, reviewable, and able to build
absolute paths via `Path(__file__).parent` rather than depending on the
caller's working directory the way repeated `--add-data` flags would. See
the comment block at the top of `citepulse.spec` for the two specific
risk areas it exists to cover (config YAML bundling, SQLite dialect
hidden import).

## Manual verification checklist

**A successful `pyinstaller` build does not prove the frozen exe actually
works.** Both risk areas below fail at *runtime*, not build time. Run this
checklist before treating any build as release-ready:

1. Run `build_dist.bat`.
2. Copy the **unzipped** `dist\citepulse\` folder to a location entirely
   outside this repo (e.g. Desktop). Testing in place risks silently
   falling through to the repo's own `.venv`/PATH and producing a false
   pass.
3. Open a new terminal. For that session only, strip this repo's
   `.venv\Scripts` from `PATH` so the test can't accidentally resolve the
   dev venv's `citepulse.exe` instead of the copied one:
   ```powershell
   $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notlike '*CitePulse*.venv*' }) -join ';'
   ```
4. From inside the copied folder, using the full path to the exe:
   - `.\citepulse.exe setup` — expect "Database initialized..." and a new
     SQLite DB under `~/.citepulse`. Proves the SQLite dialect hidden
     import actually works (not just builds).
   - `.\citepulse.exe audit https://example.com` — expect `band
     "critical"` / the llms.txt-missing remediation text. Proves the YAML
     `datas` bundling actually works.
   - `.\citepulse.exe audit https://www.hubspot.com` and
     `...stripe.com` — expect `band "best_in_class"`, matching
     `run_test.bat`'s existing sample audits. (Live network checks against
     real sites — if a result ever doesn't match, confirm the site's
     `llms.txt` hasn't simply changed before assuming a packaging bug.)
5. Confirm no admin-elevation prompt appeared at any point.
6. Note whether Windows showed an "unrecognized/unverified publisher"
   warning on first run — expected, no paid code-signing certificate, not
   a failure; record it, don't try to suppress it. Confirmed live
   (2026-09-17, Windows 10 Pro 19045): double-clicking `Start
   CitePulse.bat` from a fresh download shows an **"Open File – Security
   Warning"** dialog (Mark-of-the-Web on the downloaded `.bat`, "Unknown
   Publisher," requires clicking **Run**) — this is the older
   Attachment-Manager unsigned-file warning, not the newer blue "Windows
   protected your PC" SmartScreen screen (that one is specific to EXE
   reputation checks; a `.bat` entry point trips this one instead). A
   second, separate **Windows Defender Firewall alert** for
   `citepulse.exe` used to also appear once Streamlit started listening
   for connections — fixed as of the `--server.address=127.0.0.1` change
   below (`citepulse/cli.py`'s `_launch_streamlit()`), which forces a
   loopback-only bind instead of Streamlit's own `0.0.0.0` default, so
   this firewall prompt should no longer appear on a fresh run; re-verify
   on the next packaged build.
7. Record the tested Windows version/account in the commit/PR description.
   This is a manual, human-attested step; there's no CI automation for it
   yet (see "Not done yet" below).
8. `.\citepulse.exe ui` still works frozen — expect a browser tab opening
   against the Streamlit app with no `sys.frozen`/pip-install error.
9. On a machine that has **never run CitePulse** (or with a clean
   `%LOCALAPPDATA%\ms-playwright` cache, to simulate one) run
   `.\citepulse.exe launch` — expect a real, from-scratch Chromium
   download (proves the in-process
   `playwright.__main__.main(["install", "chromium"])` path actually
   works, not just a cache hit) followed by the browser tab opening.
10. From that same session, run a KPI #48/#58 audit (via the UI's Run
    Audit page, or `.\citepulse.exe audit <url>` with those KPIs
    selected) — expect it to actually complete (not silently render both
    KPIs unavailable), proving the bundled Playwright browser the
    previous step installed actually launches and drives a page.
11. Double-click `Start CitePulse.bat` from Windows Explorer (not a
    terminal) — expect a console window to open, then a browser tab with
    the CitePulse UI, with no visible Python/pip error text anywhere in
    the console window. Close the browser tab and confirm the console
    window's `pause` keeps it open (so a failure would have been
    readable) rather than closing instantly.
12. **Explicitly test a second, network-disconnected run** after step 9's
    Chromium install succeeds: disconnect the network (or block outbound
    access), then run `.\citepulse.exe launch` and confirm the UI itself
    still opens and Sites/History/Manage still work.

    **Precisely what this proves, and what it doesn't**: CitePulse has no
    claim to being usable fully offline — auditing any site requires
    reaching that site over the network, exactly as a browser would, and
    KPI #22/#24/#45/#62's citation-rate corpus plus competitor discovery
    do live web searches against public search engines on top of that.
    None of that works, or should be expected to work, without network
    access. What genuinely should never require network once Chromium is
    installed: the app launching, the DB opening, and the Sites/History/
    Manage pages (pure local reads/writes). This step exists to catch a
    *regression* — e.g. an accidental network call added to app startup,
    a spurious update check, telemetry — not to demonstrate the product
    can perform an audit with no network. If a KPI that needs live
    network fails offline, that's expected and not a bug; a failure in
    the UI opening or the non-network pages is the actual signal this
    step is checking for.

## Launching the UI from a packaged build

`citepulse.exe launch` is the one command a packaged zip's launcher,
`Start CitePulse.bat` (checked into the repo root, copied into
`dist\citepulse\` by `build_dist.bat` before zipping), actually runs:

```bat
@echo off
cd /d "%~dp0"
set "CITEPULSE_DATA_DIR=%~dp0data"
set "CORE_EDITION_MODE=true"
set "COMPETITOR_DISCOVERY_ENABLED=false"
citepulse.exe launch
pause
```

(v1.6.1 added the three `set` lines above — v1.6.0's launcher called
`citepulse.exe launch` directly with no env vars set, which silently left
`core_edition_mode` at its `False` default in the shipped zip. See the
Core edition section below and the fix in PR #57/v1.6.1.)

`launch` does `init_db()` → a best-effort `_ensure_playwright_chromium()`
call (never blocks the launch on failure -- Task Readiness KPIs #48/#58
just stay unavailable until it succeeds) → the same Streamlit launch
`citepulse ui` does. A non-technical zip user double-clicks the `.bat`
file and never sees a CLI, runs `setup`/`ui` separately, or needs to know
either command exists. `pause` at the end keeps the console window open
after a failed launch so the error is actually readable, instead of the
window instantly closing.

Both `citepulse.exe ui` and `citepulse.exe launch` work in a frozen
build now (previously `ui` raised `sys.frozen`-gated `ClickException`,
and `_ensure_playwright_chromium()` early-returned with a
pip-install-instead message). `citepulse.spec` bundles Streamlit and
Playwright via `PyInstaller.utils.hooks.collect_all()` — confirmed via a
throwaway pre-implementation spike, not assumed, since Streamlit+
PyInstaller is a historically fragile combination and neither package
has a PyInstaller stdhook as of `pyinstaller-hooks-contrib` 2026.7. Two
sharp edges the spike surfaced, both handled in `cli.py`:

1. Streamlit's own dev-mode auto-detection misfires when frozen
   (`RuntimeError: server.port does not work when
   global.developmentMode is true`) — fixed by setting
   `STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false` before the in-process
   `streamlit.web.cli.main()` call `_launch_streamlit()` makes when
   `sys.frozen`.
2. A frozen Playwright driver resolves its browser-cache location
   differently than a normal pip install — it looks for a
   `.local-browsers` directory next to the bundled driver rather than
   `%LOCALAPPDATA%\ms-playwright`. `cli.py` sets
   `PLAYWRIGHT_BROWSERS_PATH=0` when frozen so an install actually lands
   somewhere the frozen driver will find again on a later run, keeping
   the whole onedir folder self-contained (no dependency on a
   user-profile cache directory living outside the folder the user was
   told to keep together). Confirmed via the spike: Chromium both
   downloads and actually launches (a real `chromium.launch()` +
   `page.goto()`) from inside a PyInstaller onedir tree this way.

`_ensure_playwright_chromium()`'s frozen branch
(`_ensure_playwright_chromium_frozen()`) calls
`playwright.__main__.main()` in-process (that function takes no
arguments and reads `sys.argv` directly) on a background thread with a
join timeout, since there's no child process to timeout-kill the way the
non-frozen `subprocess.run(..., timeout=...)` path does. Same
never-raises/always-returns-`bool`/always-echoes-a-message contract as
before, preserved exactly.

Bundling Streamlit/Playwright adds real size: a build with both is
roughly 300+ MB (vs. well under 100MB before), most of it the
Node-containing `playwright/driver/` tree plus Streamlit's static
frontend assets. Expected, not a regression to chase down.

## Core edition

`settings.core_edition_mode` (default `False`) is a build/branding flag
for a minimum "v1 core" version aimed at enterprise-LAN users: 6 KPIs
(#1, #22, #24, #46, #48, #58 — Compare Models/OpenRouter/batch/
multi-model surfaces need either 3x local Ollama or cloud API keys,
neither of which fits a locked-down, no-account audience), Sites/Run
Audit/History/Manage pages only. "No account" means no CitePulse-owned
service/API key is ever required (local Ollama, no telemetry) — it does
not mean audits run without network access; auditing a site still
requires reaching that site, and citation-rate/competitor-discovery
features do live web searches. See item 12 of the manual verification
checklist above for exactly what "no internet required" does and doesn't
cover. It is read via
`get_settings()` at the specific UI call sites that branch on it — no new
function parameters, no second PyInstaller spec/build target. `Start
CitePulse.bat` sets `CORE_EDITION_MODE=true` and
`COMPETITOR_DISCOVERY_ENABLED=false` as environment variables before
calling `citepulse.exe launch` (pydantic-settings reads env vars
case-insensitively with no prefix, so this is sufficient — no `.env` file
needed in the zip); a normal pip install/dev checkout, or the exe run
directly without going through the `.bat`, defaults to `False` (today's
full nav, unchanged). See `CLAUDE.md`'s Status paragraph for the exact
list of what it gates.

`Start CitePulse.bat` also sets `CITEPULSE_DATA_DIR=%~dp0data`, so each
extracted copy of the zip keeps its own self-contained `citepulse.db`/
logs next to the exe rather than sharing the default
`%USERPROFILE%\.citepulse` location across every CitePulse install (dev
checkout, older build, or another extracted zip) on the same machine —
that shared default is still what running `citepulse.exe` directly
(bypassing the `.bat`) or a normal pip install uses.

This is a flag, not a fork or a separate build pipeline: `build_dist.bat`
/ `citepulse.spec` are unchanged by it — the same onedir build works for
both editions, and which one a given zip behaves like depends only on
whatever `.env`/environment it ships with. There is no `--core` PyInstaller
flag or second `.spec` file to keep in sync.

## Not done yet

- macOS/Linux builds — this whole workflow is Windows-only (`.bat`-based,
  same as the rest of this repo's tooling today).
- GitHub Actions release automation — build + verification are manual/
  local for now.
- Code signing — the SmartScreen warning is a known, accepted limitation,
  not solved. A paid certificate is a separate cost/process decision.
- Ollama bundling — not needed. Ollama's own Windows installer is already
  a per-user, no-admin install (`%LOCALAPPDATA%\Programs\Ollama`).
- Auto-update mechanism.
- UPX compression — deliberately skipped: UPX-compressed binaries can
  themselves trigger more AV false positives, the opposite of what a
  locked-down-machine audience needs.
