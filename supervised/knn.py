"""K-Nearest Neighbors — Classifier and Regressor from scratch.

KNN is a non-parametric, lazy learning algorithm.
For classification: majority vote among k neighbors.
For regression: mean of k neighbors' values.

Time complexity:
  - Training: O(1) — just stores the data
  - Prediction: O(n * d) per query (brute force Euclidean)
"""
import numpy as np
from typing import Literal
from collections import Counter


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(a - b)))


class KNNClassifier:
    """K-Nearest Neighbors classifier.

    Example:
        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print(knn.score(X_test, y_test))
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ):
        self.k = k
        self.metric = metric
        self._X_train: np.ndarray = None
        self._y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training data (lazy learning — no actual training)."""
        self._X_train = np.array(X)
        self._y_train = np.array(y)
        return self

    def _distances(self, x: np.ndarray) -> np.ndarray:
        """Compute distances from query point x to all training points."""
        dist_fn = euclidean_distance if self.metric == "euclidean" else manhattan_distance
        return np.array([dist_fn(x, x_train) for x_train in self._X_train])

    def _predict_single(self, x: np.ndarray):
        """Predict class for a single sample."""
        distances = self._distances(x)
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self._y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (vote fractions).

        Returns:
            Array of shape (n_samples, n_classes)
        """
        classes = sorted(set(self._y_train))
        probas = []
        for x in X:
            distances = self._distances(x)
            k_indices = np.argsort(distances)[: self.k]
            k_labels = self._y_train[k_indices]
            counts = Counter(k_labels)
            probas.append([counts.get(c, 0) / self.k for c in classes])
        return np.array(probas)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on given data."""
        return float(np.mean(self.predict(X) == y))


class KNNRegressor:
    """K-Nearest Neighbors regressor (mean of k neighbors).

    Example:
        knn = KNNRegressor(k=5)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print(knn.score(X_test, y_test))  # R²
    """

    def __init__(
        self,
        k: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ):
        self.k = k
        self.metric = metric
        self.weights = weights
        self._X_train: np.ndarray = None
        self._y_train: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """Store training data."""
        self._X_train = np.array(X)
        self._y_train = np.array(y, dtype=float)
        return self

    def _predict_single(self, x: np.ndarray) -> float:
        dist_fn = euclidean_distance if self.metric == "euclidean" else manhattan_distance
        distances = np.array([dist_fn(x, xt) for xt in self._X_train])
        k_indices = np.argsort(distances)[: self.k]
        k_distances = distances[k_indices]
        k_values = self._y_train[k_indices]

        if self.weights == "distance":
            # Inverse distance weighting (handle zero distance)
            k_distances = np.where(k_distances == 0, 1e-10, k_distances)
            w = 1.0 / k_distances
            return float(np.average(k_values, weights=w))
        return float(np.mean(k_values))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values for X."""
        return np.array([self._predict_single(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score on given data."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error."""
        return float(np.mean((self.predict(X) - y) ** 2))


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    np.random.seed(42)

    print("=" * 60)
    print("KNN Classifier — Iris Dataset (150 samples, 3 classes)")
    print("=" * 60)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for k in [1, 3, 5, 7]:
        our = KNNClassifier(k=k).fit(X_train, y_train)
        sk = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        print(f"  k={k}: ours={our.score(X_test, y_test):.4f}, sklearn={sk.score(X_test, y_test):.4f}")

    print("\n" + "=" * 60)
    print("KNN Regressor — Diabetes Dataset")
    print("=" * 60)
    diabetes = load_diabetes()
    X2, y2 = diabetes.data, diabetes.target
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    scaler2 = StandardScaler()
    X_train2 = scaler2.fit_transform(X_train2)
    X_test2 = scaler2.transform(X_test2)

    for k in [3, 5, 10]:
        our = KNNRegressor(k=k, weights="distance").fit(X_train2, y_train2)
        sk = KNeighborsRegressor(n_neighbors=k, weights="distance").fit(X_train2, y_train2)
        print(f"  k={k}: ours R²={our.score(X_test2, y_test2):.4f}, sklearn R²={sk.score(X_test2, y_test2):.4f}")
