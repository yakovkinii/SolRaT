import unittest
import yatools.logging_config

if __name__ == "__main__":
    yatools.logging_config.init()
    loader = unittest.TestLoader()
    start_dir = "entry_points/"
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner()
    runner.run(suite)
