"""End-to-end exercise of citepulse.ui.components.render_model_picker's
pull flow via streamlit.testing.v1.AppTest, rendered through the Run
Audit page (citepulse/ui/pages/run_audit.py) since that's where it's
wired in. Ollama's /api/tags and /api/pull are mocked with respx against
a small piece of shared mutable state, so the mock behaves like a real
local Ollama instance that actually gained a model after a pull --
proving the "select Pull, then Confirm pull, then it's installed and
selectable without a page reload" flow end to end, not just that
pull_model() itself parses NDJSON correctly (that's covered directly in
tests/test_ai_engines/test_ollama.py). Skipped when the optional `ui`
extra (streamlit) isn't installed, same as the other UI test files."""

import json
from pathlib import Path

import pytest
import respx
from httpx import Response

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import citepulse.db as db_module  # noqa: E402
import citepulse.model_recommender as model_recommender_module  # noqa: E402
import citepulse.settings as settings_module  # noqa: E402

_PAGE_PATH = str(
    Path(__file__).parent.parent / "citepulse" / "ui" / "pages" / "run_audit.py"
)


def _reset_singletons(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)
    monkeypatch.setattr(db_module, "_engine", None)


def _ndjson(*lines: dict) -> str:
    return "\n".join(json.dumps(line) for line in lines)


@respx.mock
def test_pull_flow_requires_two_clicks_then_installs_and_becomes_selectable(
    monkeypatch, tmp_path
):
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    # Deterministic ranking regardless of the test machine's real RAM --
    # list_installed_models() itself is left un-mocked so it genuinely
    # goes through the (respx-mocked) HTTP call below.
    monkeypatch.setattr(
        model_recommender_module, "detect_available_ram_gb", lambda: 128.0
    )

    installed_state: list[str] = []

    def _tags_handler(request):
        return Response(200, json={"models": [{"name": n} for n in installed_state]})

    respx.get("http://localhost:11434/api/tags").mock(side_effect=_tags_handler)

    pull_requests = []

    def _pull_handler(request):
        pull_requests.append(json.loads(request.content))
        installed_state.append("llama3.1:8b")
        body = _ndjson(
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 500, "total": 1000},
            {"status": "success"},
        )
        return Response(200, text=body)

    respx.post("http://localhost:11434/api/pull").mock(side_effect=_pull_handler)

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert any("No installed Ollama models" in w.value for w in at.warning)
    pull_button = next(b for b in at.button if b.label.startswith("Pull llama3.1:8b"))

    # First click only opens the two-step confirm -- no network call yet.
    pull_button.click()
    at.run()

    assert pull_requests == []
    assert any(b.label == "Confirm pull" for b in at.button)

    confirm_button = next(b for b in at.button if b.label == "Confirm pull")
    confirm_button.click()
    at.run()

    assert not at.exception
    assert len(pull_requests) == 1
    assert pull_requests[0] == {"name": "llama3.1:8b"}
    assert any("is now installed" in s.value for s in at.success)
    # The freshly pulled model is now selectable -- proof the picker
    # refreshed itself (a script rerun) rather than needing the user to
    # reload the page.
    assert any("llama3.1:8b" in opt for opt in at.selectbox[0].options)
    assert not any(b.label.startswith("Pull llama3.1:8b") for b in at.button)


@respx.mock
def test_pull_button_alone_never_triggers_a_download(monkeypatch, tmp_path):
    """A single click on "Pull ..." must never itself call /api/pull --
    only the second, explicit "Confirm pull" click may (see the scope
    note: this is a real multi-GB network download the user should not
    trigger by accident)."""
    _reset_singletons(monkeypatch, tmp_path)
    db_module.init_db()
    monkeypatch.setattr(
        model_recommender_module, "detect_available_ram_gb", lambda: 128.0
    )
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=Response(200, json={"models": []})
    )
    # /api/pull is deliberately left unmocked: respx's default
    # assert-all-mocked behavior means a real call to it here would raise
    # inside the script, surfacing as at.exception -- the cleanest proof
    # a single "Pull" click alone never reaches pull_model().

    at = AppTest.from_file(_PAGE_PATH, default_timeout=30)
    at.run()

    pull_button = next(b for b in at.button if b.label.startswith("Pull llama3.1:8b"))
    pull_button.click()
    at.run()

    assert not at.exception
