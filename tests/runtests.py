import unittest
loader = unittest.TestLoader()
tests = loader.discover('tests')
runner = unittest.TextTestRunner()
runner.run(tests)
