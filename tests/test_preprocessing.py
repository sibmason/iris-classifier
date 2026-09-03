import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train

# Basic sanity check: test data should have at least one sample and one feature.
X_test, y_test, target_names = train.load_data()

assert X_test.ndim == 2, "X_test should be a 2D array"
assert y_test.ndim == 1, "y_test should be a 1D array"
assert X_test.shape[0] > 0, "X_test should contain samples"
assert X_test.shape[1] > 0, "X_test should contain features"
print("Data loaded successfully;", X_test.shape)
print("Target names:",target_names)
