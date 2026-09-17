"""Streamlit entry point -- launched via `citepulse ui` (citepulse/cli.py),
which runs `streamlit run` on this file. Not imported by the CLI/pytest
otherwise, since it requires the optional `ui` extra (streamlit)."""

from pathlib import Path

import streamlit as st

from citepulse.db import init_db
from citepulse.logging_setup import configure_logging
from citepulse.settings import get_settings

st.set_page_config(page_title="CitePulse", page_icon="📡", layout="wide")

# Both idempotent and safe to call on every rerun (configure_logging() by
# log-file path, init_db() via SQLModel.metadata.create_all).
configure_logging()
init_db()

_PAGES_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page(str(_PAGES_DIR / "sites.py"), title="Sites", icon=":material/language:"),
    st.Page(
        str(_PAGES_DIR / "run_audit.py"),
        title="Run audit",
        icon=":material/play_arrow:",
        default=True,
    ),
    st.Page(str(_PAGES_DIR / "history.py"), title="History", icon=":material/history:"),
]

# Compare Models triples LLM cost/runtime (3x local Ollama or OpenRouter
# cloud keys) -- a core-edition build (see settings.core_edition_mode)
# omits it from the nav entirely rather than rendering-and-disabling it,
# so there's no dead/confusing nav entry and no code path inside
# compare.py to keep alive for a build that never shows it. The page's
# code itself is untouched and still reachable in a non-core build.
if not get_settings().core_edition_mode:
    pages.append(
        st.Page(
            str(_PAGES_DIR / "compare.py"),
            title="Compare models",
            icon=":material/compare_arrows:",
        )
    )

pages.append(
    st.Page(
        str(_PAGES_DIR / "manage.py"),
        title="Manage",
        icon=":material/tune:",
    )
)

st.navigation(pages).run()
