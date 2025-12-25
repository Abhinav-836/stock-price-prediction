"""
Test Suite for Stock Price Prediction Project

This package contains all unit tests for the project.

Test Modules:
- test_preprocess.py: Tests for data preprocessing functions
- test_models.py: Tests for model training and evaluation
- run_tests.py: Main test runner

Usage:
    # Run all tests
    python tests/run_tests.py
    
    # Run specific test module
    python -m unittest tests.test_preprocess
    python -m unittest tests.test_models
    
    # Run specific test class
    python -m unittest tests.test_preprocess.TestPreprocessing
    
    # Run specific test method
    python -m unittest tests.test_preprocess.TestPreprocessing.test_create_date_features
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Import test classes for easier access
from .test_preprocess import TestPreprocessing, TestDataIntegrity
from .test_model import TestModelTraining, TestModelEvaluation, TestFeaturePreparation

__all__ = [
    'TestPreprocessing',
    'TestDataIntegrity', 
    'TestModelTraining',
    'TestModelEvaluation',
    'TestFeaturePreparation'
]