import functools
import inspect
import logging
import threading
import time

VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

LOGGING_LEVEL = VERBOSE
_state = threading.local()


def _get_level() -> int:
    return getattr(_state, "level", 0)


def _set_level(value: int) -> None:
    _state.level = value


def _log_call_start(name: str, level: int, levelno: int, source_file: str, line_number: int) -> None:
    ident = "⋅ " * level
    logger = logging.getLogger()
    record = logger.makeRecord(
        logger.name,
        levelno,
        source_file,
        line_number,
        ident + name,
        {},
        None,
        "",
    )
    logger.handle(record)


def _log_call_end(message: str, level: int, source_file: str, line_number: int) -> None:
    ident = "⋅ " * level
    logger = logging.getLogger()
    record = logger.makeRecord(
        logger.name,
        LOGGING_LEVEL,
        source_file,
        line_number,
        ident + message,
        {},
        None,
        "",
    )
    logger.handle(record)


def log_method(method):
    @functools.wraps(method)
    def decorator(self, *args, **kwargs):
        class_name = self.__class__.__name__
        source_file = inspect.getsourcefile(method)
        line_number = inspect.getsourcelines(method)[1]

        level = _get_level()
        _log_call_start(f"{class_name}.{method.__name__}", level, LOGGING_LEVEL, source_file, line_number)

        _set_level(level + 1)
        start_time = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            end_time = time.perf_counter()
            _set_level(level)
            _log_call_end(f"{end_time - start_time:.4f}s", level, source_file, line_number)

    return decorator


def log_method_experimental(method):  # pragma: no cover
    @functools.wraps(method)
    def decorator(self, *args, **kwargs):
        class_name = self.__class__.__name__
        source_file = inspect.getsourcefile(method)
        line_number = inspect.getsourcelines(method)[1]

        level = _get_level()
        _log_call_start(
            f"{class_name}.{method.__name__} (experimental feature, use with caution!)",
            level,
            logging.WARNING,
            source_file,
            line_number,
        )

        _set_level(level + 1)
        start_time = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            end_time = time.perf_counter()
            _set_level(level)
            _log_call_end(f"{end_time - start_time:.4f}s", level, source_file, line_number)

    return decorator


def log_function(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        source_file = inspect.getsourcefile(function)
        line_number = inspect.getsourcelines(function)[1]

        level = _get_level()
        _log_call_start(function.__name__, level, LOGGING_LEVEL, source_file, line_number)

        _set_level(level + 1)
        start_time = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            _set_level(level)
            _log_call_end(
                f"{function.__name__} finished in {end_time - start_time:.6f}s",
                level,
                source_file,
                line_number,
            )

    return decorator


def log_function_experimental(function):  # pragma: no cover
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        source_file = inspect.getsourcefile(function)
        line_number = inspect.getsourcelines(function)[1]

        level = _get_level()
        _log_call_start(
            f"{function.__name__} (experimental feature, use with caution!)",
            level,
            logging.WARNING,
            source_file,
            line_number,
        )

        _set_level(level + 1)
        start_time = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            _set_level(level)
            _log_call_end(
                f"{function.__name__} finished in {end_time - start_time:.6f}s",
                level,
                source_file,
                line_number,
            )

    return decorator
