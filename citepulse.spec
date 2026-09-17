# PyInstaller spec for the standalone Windows build (see docs/PACKAGING.md).
#
# Generated starting from `pyinstaller --name citepulse citepulse/cli.py`
# then hand-edited for the codebase-specific requirements below -- if
# PyInstaller's bundled version changes and you regenerate this file, port
# them all back in rather than dropping them:
#
# 1. `datas`: several modules (citepulse/remediation.py,
#    citepulse/business_context.py, citepulse/reporting.py,
#    citepulse/model_recommender.py) load citepulse/config/*.yaml via
#    `Path(__file__).parent / "config" / ...` at runtime. Every file
#    under citepulse/config/ must be bundled at that same relative
#    subpath or the corresponding feature crashes/silently degrades in a
#    frozen build (template rendering, priority scoring, or the model
#    picker's catalogs). If a new YAML file is ever added under
#    citepulse/config/, add it here too.
# 2. `hiddenimports`: SQLAlchemy's SQLite dialect plugin discovery is a
#    known PyInstaller pitfall -- the build succeeds but `citepulse
#    setup`/`audit` fail at runtime with a dialect-not-found error. bs4/
#    lxml are declared dependencies not yet imported by any wired KPI
#    (only #46 is wired, and it doesn't use them) -- included defensively
#    so the next KPI that does use them doesn't silently break only in the
#    frozen build.
# 3. Streamlit bundling (`core_edition_mode`/"v1 core" packaging): pulled
#    in via `collect_all("streamlit")` rather than hand-listing files, so
#    its bundled static frontend assets (streamlit/static/) and its many
#    dynamically-imported submodules come along automatically -- confirmed
#    against a throwaway spike (see docs/PACKAGING.md) that a hand-rolled
#    hiddenimports list is not reliable here, and that no PyInstaller
#    stdhook covers streamlit as of pyinstaller-hooks-contrib 2026.7 (only
#    `hook-anyio.py`/`hook-uvicorn.py`/`hook-websockets.py`, streamlit's
#    own transitive deps, exist there). `citepulse/ui/app.py` and
#    `citepulse/ui/pages/*.py` are CitePulse's own files, not Streamlit's,
#    so collect_all("streamlit") does not pick them up -- bundled via an
#    explicit `datas` entry below instead.
# 4. Playwright bundling (`core_edition_mode`): `collect_all("playwright")`
#    pulls in `playwright/driver/` -- a non-Python, Node-containing tree
#    (a real, confirmed size increase: ~120MB just for the driver package
#    before any browser is installed) -- which resolves correctly from
#    inside a PyInstaller onedir tree per the same spike. Two real,
#    confirmed-via-spike sharp edges callers must handle, not this spec:
#    (a) Streamlit's own dev-mode auto-detection misfires when frozen
#    (`RuntimeError: server.port does not work when global.developmentMode
#    is true`) unless `STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false` is set
#    before `streamlit.web.cli.main()` runs -- see cli.py's
#    `_launch_streamlit()`; (b) Playwright's browser-cache path resolution
#    differs when frozen (it looks for a `.local-browsers` directory next
#    to the bundled driver rather than the normal `%LOCALAPPDATA%\
#    ms-playwright`), so cli.py sets `PLAYWRIGHT_BROWSERS_PATH=0` when
#    frozen so an install actually lands somewhere the frozen driver will
#    find it again on a later run.
#
# Builds as onedir (a folder), not onefile: onedir starts faster on every
# run and is less likely to trip AV/SmartScreen heuristics on a locked-
# down corporate machine than onefile's self-extract-to-temp-dir-on-every-
# launch behavior.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

repo_root = Path(SPECPATH)

datas = [
    (str(repo_root / "citepulse" / "config" / "remediation.yaml"), "citepulse/config"),
    (str(repo_root / "citepulse" / "config" / "standard_kpi_mapping.yaml"), "citepulse/config"),
    (str(repo_root / "citepulse" / "config" / "model_catalog.yaml"), "citepulse/config"),
    (str(repo_root / "citepulse" / "config" / "openrouter_catalog.yaml"), "citepulse/config"),
    (str(repo_root / "citepulse" / "config" / "scoring_bands.yaml"), "citepulse/config"),
    (str(repo_root / "citepulse" / "ui" / "app.py"), "citepulse/ui"),
    (str(repo_root / "citepulse" / "ui" / "pages"), "citepulse/ui/pages"),
]
# .streamlit/config.toml lives at the repo root today (theme colors used
# by citepulse/ui/components.py's dark/light-mode multiselect-tag CSS) --
# Streamlit resolves it relative to the process's current working
# directory, which for a packaged build is dist\citepulse\ itself (see
# "Start CitePulse.bat"'s `cd /d "%~dp0"`), so it's bundled at the onedir
# root here, not nested under citepulse/ui, to match that same layout.
if (repo_root / ".streamlit").exists():
    datas.append((str(repo_root / ".streamlit"), ".streamlit"))

binaries = []

hiddenimports = [
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.dialects.sqlite.pysqlite",
    "bs4",
    "bs4.builder._lxml",
    "lxml.etree",
    "lxml.html",
]

for _pkg in ("streamlit", "playwright"):
    _datas, _binaries, _hiddenimports = collect_all(_pkg)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hiddenimports

# citepulse.cli (the Analysis entry point) never itself imports
# citepulse.ui.* at module scope -- Streamlit only loads
# citepulse/ui/app.py and citepulse/ui/pages/*.py by *file path*
# (st.Page(str(path)), read from disk and exec'd -- see the `datas`
# entries above), never via a Python `import` statement, so PyInstaller's
# static import graph never discovers citepulse.ui.components on its own.
# Confirmed via a real frozen-build test during this feature's own
# verification: without this, `citepulse.exe ui` served an HTTP response
# but crashed with `ModuleNotFoundError: No module named
# 'citepulse.ui.components'` the moment a page's own `from citepulse.ui.
# components import ...` statement ran (every page under citepulse/ui/
# pages/ imports from this one shared module, nothing else under
# citepulse/ui/). Explicit hiddenimport rather than
# collect_submodules("citepulse.ui") on purpose -- that would also try to
# statically import the page scripts themselves, which make bare
# top-level Streamlit calls (st.subheader(), render_header(), ...) meant
# to run inside a live Streamlit script context, not at plain import
# time; the page *files* still need their separate `datas` entries above
# too, since Streamlit reads those by path, not import.
hiddenimports.append("citepulse.ui.components")

a = Analysis(
    [str(repo_root / "citepulse" / "cli.py")],
    pathex=[str(repo_root)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="citepulse",
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="citepulse",
)
