import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train

# Train the model and check the accuracy meets the threshold.
accuracy = train.train_and_evaluate()
assert accuracy >= 0.9, f"Model accuracy {accuracy:.3f} is below expected threshold"
print("Done")
