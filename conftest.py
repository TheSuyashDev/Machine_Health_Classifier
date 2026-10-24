# conftest.py — pytest configuration for the CNC Machine Health Classifier project.
# This file ensures that the project root is added to sys.path so that
# `from src.data_preprocessing import ...` works correctly from the tests/ folder.

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
