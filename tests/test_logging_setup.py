import logging

import citepulse.settings as settings_module
from citepulse.logging_setup import configure_logging


def _reset_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("CITEPULSE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings_module, "_settings", None)


def _reset_citepulse_logger():
    logger = logging.getLogger("citepulse")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_configure_logging_creates_log_file_and_writes_to_it(monkeypatch, tmp_path):
    _reset_settings(monkeypatch, tmp_path)
    _reset_citepulse_logger()

    configure_logging()

    log_file = tmp_path / "logs" / "citepulse.log"
    assert log_file.exists()

    logging.getLogger("citepulse.some_module").warning("test message")

    assert "test message" in log_file.read_text(encoding="utf-8")


def test_configure_logging_respects_explicit_level(monkeypatch, tmp_path):
    _reset_settings(monkeypatch, tmp_path)
    _reset_citepulse_logger()

    configure_logging(level="DEBUG")

    log_file = tmp_path / "logs" / "citepulse.log"
    logging.getLogger("citepulse.some_module").debug("debug-level message")

    assert "debug-level message" in log_file.read_text(encoding="utf-8")


def test_configure_logging_twice_does_not_duplicate_handlers(monkeypatch, tmp_path):
    _reset_settings(monkeypatch, tmp_path)
    _reset_citepulse_logger()

    configure_logging()
    handlers_after_first = len(logging.getLogger("citepulse").handlers)
    configure_logging()
    handlers_after_second = len(logging.getLogger("citepulse").handlers)

    assert handlers_after_first == handlers_after_second == 2


def test_configured_logger_does_not_propagate_to_root(monkeypatch, tmp_path):
    _reset_settings(monkeypatch, tmp_path)
    _reset_citepulse_logger()

    configure_logging()

    assert logging.getLogger("citepulse").propagate is False


def test_a_later_call_with_a_lower_level_still_takes_effect(monkeypatch, tmp_path):
    """The idempotent (no-duplicate-handler) fast path must still update
    the existing file handler's level -- otherwise a later
    configure_logging("DEBUG") after an earlier configure_logging("INFO")
    against the same log file would be silently ineffective."""
    _reset_settings(monkeypatch, tmp_path)
    _reset_citepulse_logger()

    configure_logging(level="INFO")
    configure_logging(level="DEBUG")

    log_file = tmp_path / "logs" / "citepulse.log"
    logging.getLogger("citepulse.some_module").debug("debug after override")

    assert "debug after override" in log_file.read_text(encoding="utf-8")
