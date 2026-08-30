import argparse
import os
from xml.parsers.expat import model
from zipfile import Path
try:
    from sklearn.datasets import load_iris
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required; install it with `pip install scikit-learn`."
    ) from exc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    X, y, target_names = load_data()
    print("Data loaded successfully;", X.shape)
    print("Target names:", target_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    model = decision_tree(X_train, y_train, args)
    
    y_pred = predict(model, X_test)
    print("Predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    accuracy = prediction_accuracy(y_test, y_pred)
    print("Accuracy:",accuracy)
    new_dir = create_output_dir()
    plot_ConfusionMatrix(y_test, y_pred, target_names, new_dir)


def create_output_dir():
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    new_dir = project_root / "outputs" / "confusion"
    new_dir.mkdir(exist_ok=True)
    return new_dir


def plot_ConfusionMatrix(y_test, y_pred, target_names, new_dir):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cmatrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = target_names)
    cmatrix_display.plot()
    plt.savefig(new_dir / "_matrix.png")
    plt.close()


def prediction_accuracy(y_test, y_pred):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def predict(model,X_test):
    return model.predict(X_test)

def decision_tree(X_train, y_train, args):
    model = DecisionTreeClassifier(random_state=args.random_state)
    print("Model initialised.")
    model.fit(X_train, y_train)
    print("Model trained.")
    return model

def load_data():
    iris = load_iris()
    A = iris.data 
    b = iris.target 
    target_names = iris.target_names
    return A, b, target_names

if __name__ == "__main__":
    main()
    