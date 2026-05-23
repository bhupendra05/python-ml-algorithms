"""Logistic Regression from scratch — binary classification with sigmoid.

Loss function: Binary Cross-Entropy
L = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))

Gradient:
dw = (1/n) * X^T (sigma(Xw+b) - y)
db = (1/n) * sum(sigma(Xw+b) - y)
"""
import numpy as np
from typing import Optional


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


class LogisticRegression:
    """Logistic Regression for binary classification.

    Uses gradient descent on binary cross-entropy loss with optional
    L2 regularization (ridge).

    Example:
        model = LogisticRegression(learning_rate=0.1, epochs=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        l2_lambda: float = 0.0,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.losses: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Train with gradient descent on binary cross-entropy.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Binary labels {0, 1}, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for epoch in range(self.epochs):
            # Forward pass
            logits = X @ self.weights + self.bias
            y_pred = sigmoid(logits)

            # Binary cross-entropy loss (with L2 regularization)
            eps = 1e-9  # numerical stability
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            if self.l2_lambda > 0:
                loss += (self.l2_lambda / (2 * n_samples)) * np.sum(self.weights**2)
            self.losses.append(loss)

            # Gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error + (self.l2_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d}/{self.epochs} — Loss: {loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities P(y=1|X).

        Returns:
            Probabilities in [0, 1], shape (n_samples,)
        """
        if self.weights is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels {0, 1}.

        Args:
            X: Feature matrix
            threshold: Decision boundary (default 0.5)

        Returns:
            Binary predictions, shape (n_samples,)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy: fraction of correctly classified samples."""
        return float(np.mean(self.predict(X) == y))

    def log_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss on given data."""
        p = self.predict_proba(X)
        eps = 1e-9
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    # --- Simple linearly separable data ---
    print("=" * 60)
    print("Binary classification — synthetic linearly separable data")
    print("=" * 60)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(learning_rate=0.1, epochs=500, verbose=True)
    model.fit(X_train, y_train)
    print(f"\nOur model accuracy:  {model.score(X_test, y_test):.4f}")
    print(f"Our model log-loss:  {model.log_loss(X_test, y_test):.4f}")

    try:
        from sklearn.linear_model import LogisticRegression as SklearnLR

        sk = SklearnLR(max_iter=1000)
        sk.fit(X_train, y_train)
        print(f"sklearn accuracy:    {sk.score(X_test, y_test):.4f}")
    except ImportError:
        pass

    # --- Real dataset: Breast Cancer ---
    print("\n" + "=" * 60)
    print("Breast Cancer dataset (569 samples, 30 features)")
    print("=" * 60)
    data = load_breast_cancer()
    X2, y2 = data.data, data.target
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )

    scaler2 = StandardScaler()
    X_train2 = scaler2.fit_transform(X_train2)
    X_test2 = scaler2.transform(X_test2)

    model2 = LogisticRegression(learning_rate=0.1, epochs=1000, l2_lambda=0.01)
    model2.fit(X_train2, y_train2)
    print(f"Our accuracy (L2 reg): {model2.score(X_test2, y_test2):.4f}")

    sk2 = SklearnLR(C=100, max_iter=1000)
    sk2.fit(X_train2, y_train2)
    print(f"sklearn accuracy:      {sk2.score(X_test2, y_test2):.4f}")
