"""Local-only observability logging -- a rotating file under CitePulse's
own data dir, and nothing else. No network handler exists anywhere in
this module by design: CitePulse never phones home (see CLAUDE.md), and
that must hold for logging too.

Configures only the "citepulse" logger (not the root logger), so every
existing `logging.getLogger("citepulse.*")` call already in the codebase
(citepulse/task_readiness/runner.py, harness.py, task_generator.py,
citepulse/crawler/search.py) picks this up for free via the logger
hierarchy, with zero changes to those files.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from citepulse.settings import get_settings

_LOGGER_NAME = "citepulse"
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Attaches a RotatingFileHandler (at `level`, or settings.log_level
    if `level` is None) writing to <data_dir>/logs/citepulse.log, plus a
    stderr handler fixed at WARNING so normal CLI/report stdout output
    stays clean regardless of the configured level.

    Idempotent by log-file path rather than a one-shot flag: re-checks
    whether the logger already has a RotatingFileHandler pointed at the
    *current* data dir's log file before reattaching. This matters
    because citepulse/ui/app.py's module body re-executes on every
    Streamlit rerun (a one-shot boolean would leak handlers or stop
    reconfiguring), and because tests give each test its own data dir via
    CITEPULSE_DATA_DIR (path-based idempotency needs no test-teardown
    hook to stay correct)."""
    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()

    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "citepulse.log"
    target_path = str(log_file.resolve())

    logger = logging.getLogger(_LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(resolved_level)

    existing_file_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and handler.baseFilename == target_path
        ),
        None,
    )
    if existing_file_handler is not None:
        # Still honor a later call requesting a different level against
        # the same log file -- only the handler *attachment* is
        # idempotent, not the level, so e.g. configure_logging("DEBUG")
        # after an earlier configure_logging("INFO") isn't silently
        # ineffective.
        existing_file_handler.setLevel(resolved_level)
        return

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(resolved_level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)
