"""Linear Regression from scratch using gradient descent.

Implements ordinary least squares via batch gradient descent with MSE loss.
Compared against sklearn LinearRegression at the end.
"""
import numpy as np
from typing import Optional


class LinearRegression:
    """Linear Regression with gradient descent optimization.

    Minimizes Mean Squared Error: L = (1/2n) * sum((y_hat - y)^2)
    Update rule: w -= lr * (1/n) * X^T (Xw - y)
                 b -= lr * (1/n) * sum(Xw - y)

    Attributes:
        weights: Coefficient vector, shape (n_features,)
        bias: Intercept scalar
        losses: MSE loss per epoch (for plotting convergence)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.losses: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Train the model using batch gradient descent.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)

        Returns:
            self (for method chaining)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Compute MSE loss
            loss = np.mean(error**2) / 2
            self.losses.append(loss)

            # Gradients
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d}/{self.epochs} — Loss: {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        if self.weights is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² (coefficient of determination).

        R² = 1 - SS_res / SS_tot
        R² = 1.0 means perfect prediction.

        Args:
            X: Feature matrix
            y: True targets

        Returns:
            R² score in (-inf, 1.0]
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean Squared Error on given data."""
        return float(np.mean((self.predict(X) - y) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean Absolute Error on given data."""
        return float(np.mean(np.abs(self.predict(X) - y)))


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    # --- Single feature ---
    print("=" * 60)
    print("Single feature linear regression (y = 3x + 5 + noise)")
    print("=" * 60)
    X = np.random.randn(200, 1)
    y = 3 * X.squeeze() + 5 + np.random.randn(200) * 0.5

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(learning_rate=0.1, epochs=500, verbose=True)
    model.fit(X_train, y_train)

    print(f"\nOur model:")
    print(f"  Weight:  {model.weights[0]:.4f} (true: 3.0)")
    print(f"  Bias:    {model.bias:.4f} (true: 5.0)")
    print(f"  R²:      {model.score(X_test, y_test):.4f}")
    print(f"  MSE:     {model.mse(X_test, y_test):.4f}")

    # Compare with sklearn
    try:
        from sklearn.linear_model import LinearRegression as SklearnLR

        sk_model = SklearnLR()
        sk_model.fit(X_train, y_train)
        print(f"\nsklearn:")
        print(f"  Weight:  {sk_model.coef_[0]:.4f}")
        print(f"  Bias:    {sk_model.intercept_:.4f}")
        print(f"  R²:      {sk_model.score(X_test, y_test):.4f}")
    except ImportError:
        print("sklearn not installed; skipping comparison")

    # --- Multiple features ---
    print("\n" + "=" * 60)
    print("Multi-feature linear regression (Boston-style dataset)")
    print("=" * 60)
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X_full = data.data[:1000]  # 1000 samples for speed
    y_full = data.target[:1000]

    # Normalize features (critical for gradient descent)
    X_norm = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-8)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_norm, y_full, test_size=0.2, random_state=42
    )

    model2 = LinearRegression(learning_rate=0.1, epochs=1000)
    model2.fit(X_train2, y_train2)
    print(f"  Our R²:     {model2.score(X_test2, y_test2):.4f}")
    print(f"  Our MSE:    {model2.mse(X_test2, y_test2):.4f}")

    sk = SklearnLR()
    sk.fit(X_train2, y_train2)
    print(f"  sklearn R²: {sk.score(X_test2, y_test2):.4f}")
