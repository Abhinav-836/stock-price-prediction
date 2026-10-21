"""
Test Suite for Stock Price Prediction Project - FIXED

Fix: this file used to import every test class at package import time. When any test module
failed to import (they all did), the whole `tests` package became unimportable, so even
`python -m unittest tests.test_preprocess` crashed. unittest discovery finds tests by itself.

Usage (from the project root):
    python tests/run_tests.py                       # all tests
    python tests/run_tests.py --coverage            # with coverage (pip install coverage)
    python tests/run_tests.py --class TestForecasting
    python -m unittest tests.test_preprocess -v
    python -m unittest tests.test_model.TestForecasting.test_band_widens_with_horizon
"""

__version__ = '1.1.0'