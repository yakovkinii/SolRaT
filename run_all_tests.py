import unittest
import warnings

import yatools.logging_config

# for coverage:
# coverage run -m unittest discover
# coverage html


if __name__ == "__main__":
    warnings.filterwarnings("error", category=DeprecationWarning)  # Raise error on DeprecationWarning
    yatools.logging_config.init()
    loader = unittest.TestLoader()
    start_dir = "tests/"
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)
