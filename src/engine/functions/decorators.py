"""
TODO
TODO  This file needs improved documentation.
TODO
"""

import inspect
import logging
import time

LOGGING_LEVEL = logging.INFO
level = 0


def log_method(method):
    """
    A decorator to log the name of a class method when it is executed.
    """

    def decorator(self, *args, **kwargs):
        global level
        class_name = self.__class__.__name__
        ident = "⋅ " * level

        logger = logging.getLogger()
        source_file = inspect.getsourcefile(method)
        line_number = inspect.getsourcelines(method)[1]
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{class_name}.{method.__name__}",
            {},
            None,
            "",
        )
        logger.handle(lr)

        level += 1
        if args or kwargs:
            start_time = time.perf_counter()
            result = method(self, *args, **kwargs)
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            result = method(self)
            end_time = time.perf_counter()
        level -= 1
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{end_time - start_time:.4f}s",
            {},
            None,
            "",
        )
        logger.handle(lr)
        return result

    return decorator


def log_method_experimental(method):
    """
    A decorator to log the name of a class method when it is executed.
    For marking the experimental features.
    """

    def decorator(self, *args, **kwargs):
        global level
        class_name = self.__class__.__name__
        ident = "⋅ " * level

        logger = logging.getLogger()
        source_file = inspect.getsourcefile(method)
        line_number = inspect.getsourcelines(method)[1]
        lr = logger.makeRecord(
            logger.name,
            logging.WARNING,
            source_file,
            line_number,
            ident + f"{class_name}.{method.__name__} (experimental feature, use with caution!)",
            {},
            None,
            "",
        )
        logger.handle(lr)

        level += 1
        if args or kwargs:
            start_time = time.perf_counter()
            result = method(self, *args, **kwargs)
            end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            result = method(self)
            end_time = time.perf_counter()
        level -= 1
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{end_time - start_time:.4f}s",
            {},
            None,
            "",
        )
        logger.handle(lr)
        return result

    return decorator


def log_function(function):
    """
    A decorator to log the name of a function when it is executed.
    """

    def decorator(*args, **kwargs):
        global level
        ident = "⋅ " * level

        logger = logging.getLogger()
        source_file = inspect.getsourcefile(function)
        line_number = inspect.getsourcelines(function)[1]
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{function.__name__}",
            {},
            None,
            "",
        )
        logger.handle(lr)

        level += 1
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        level -= 1
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{function.__name__} finished in {end_time - start_time:.6f}s",
            {},
            None,
            "",
        )
        logger.handle(lr)

        return result

    return decorator


def log_function_experimental(function):
    """
    A decorator to log the name of a function when it is executed.
    For marking the experimental features.
    """

    def decorator(*args, **kwargs):
        global level
        ident = "⋅ " * level

        logger = logging.getLogger()
        source_file = inspect.getsourcefile(function)
        line_number = inspect.getsourcelines(function)[1]
        lr = logger.makeRecord(
            logger.name,
            logging.WARNING,
            source_file,
            line_number,
            ident + f"{function.__name__} (experimental feature, use with caution!)",
            {},
            None,
            "",
        )
        logger.handle(lr)

        level += 1
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        level -= 1
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{function.__name__} finished in {end_time - start_time:.6f}s",
            {},
            None,
            "",
        )
        logger.handle(lr)

        return result

    return decorator


def log_function_not_tested(function):  # pragma: no cover
    """
    A decorator to log the name of a function when it is executed.
    """

    def decorator(*args, **kwargs):
        global level
        ident = "⋅ " * level

        logger = logging.getLogger()
        source_file = inspect.getsourcefile(function)
        line_number = inspect.getsourcelines(function)[1]
        lr = logger.makeRecord(
            logger.name,
            logging.WARNING,
            source_file,
            line_number,
            ident + f"{function.__name__} (not tested)",
            {},
            None,
            "",
        )
        logger.handle(lr)

        level += 1
        start_time = time.perf_counter()
        result = function(*args, **kwargs)
        end_time = time.perf_counter()
        level -= 1
        lr = logger.makeRecord(
            logger.name,
            LOGGING_LEVEL,
            source_file,
            line_number,
            ident + f"{function.__name__} finished in {end_time - start_time:.4f}s",
            {},
            None,
            "",
        )
        logger.handle(lr)

        return result

    return decorator
