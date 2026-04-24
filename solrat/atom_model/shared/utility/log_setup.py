import logging

import coloredlogs

from solrat.engine.functions.decorators import VERBOSE


def setup_logging(level: int = logging.INFO) -> None:
    """
    Initialize solrat logging.

    Pass logging.INFO for normal output.
    Pass logging.DEBUG to also see engine trace messages (VERBOSE level).
    Third-party debug noise (scipy, pandas, etc.) is always suppressed.
    """
    logging.addLevelName(VERBOSE, "VERBOSE")

    # Map DEBUG to VERBOSE so third-party DEBUG messages (level 10) stay hidden.
    effective_level = VERBOSE if level == logging.DEBUG else level

    coloredlogs.DEFAULT_FIELD_STYLES = dict(
        asctime=dict(color="white"),
        hostname=dict(color="magenta"),
        levelname=dict(color="blue"),
        name=dict(color="blue"),
        programname=dict(color="cyan"),
        username=dict(color="yellow"),
        pathname=dict(color="white", faint=True),
        lineno=dict(color="white", faint=True),
    )

    coloredlogs.DEFAULT_LEVEL_STYLES = dict(
        spam=dict(color="green", faint=True),
        debug=dict(color="white", faint=True),
        verbose=dict(color="cyan", faint=True),
        info=dict(color="green", bold=True),
        notice=dict(color="magenta", bold=True),
        warning=dict(color="yellow", bold=True),
        success=dict(color="green", bold=True),
        error=dict(color="red", bold=True),
        critical=dict(color="red", bold=True),
    )

    coloredlogs.DEFAULT_LOG_LEVEL = effective_level
    coloredlogs.DEFAULT_LOG_FORMAT = "%(pathname)s:%(lineno)d\n%(asctime)s %(levelname).4s %(message)s"
    coloredlogs.DEFAULT_DATE_FORMAT = "%H:%M:%S"
    coloredlogs.install()
