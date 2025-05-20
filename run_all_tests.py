import unittest
import yatools.logging_config

if __name__ == "__main__":
    # for coverage:
    # coverage run -m unittest discover
    # coverage html
    yatools.logging_config.init()
    loader = unittest.TestLoader()
    start_dir = "tests/"
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)
