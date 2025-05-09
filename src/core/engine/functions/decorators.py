import logging


def not_tested(func):
    def wrapper(*args, **kwargs):
        logging.warning(f"Warning: {func.__name__} is not tested.")
        return func(*args, **kwargs)

    return wrapper
