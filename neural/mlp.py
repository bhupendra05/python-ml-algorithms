"""Multi-Layer Perceptron (MLP) from scratch using NumPy.

Implements:
- Configurable layer sizes
- ReLU and Sigmoid activations
- Backpropagation with chain rule
- Mini-batch gradient descent
- Binary and multi-class classification

Architecture example:
  Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
"""
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────── Activations ─────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_derivative), "relu": (relu, relu_derivative)}


# ─────────────────────────── MLP ────────────────────────────────────

class MLP:
    """Multi-Layer Perceptron (fully-connected neural network).

    Supports:
    - Arbitrary depth and width
    - ReLU or Sigmoid hidden activations
    - Binary classification (sigmoid output + BCE loss)
    - Multi-class classification (softmax output + cross-entropy)
    - He initialization (ReLU) or Xavier initialization (Sigmoid)
    - Mini-batch gradient descent

    Example (binary classification):
        mlp = MLP(layer_sizes=[20, 64, 32, 1], activation="relu", task="binary")
        mlp.fit(X_train, y_train, epochs=100, batch_size=32, lr=0.01)
        print(mlp.score(X_test, y_test))

    Example (multi-class):
        mlp = MLP(layer_sizes=[64, 128, 64, 10], activation="relu", task="multiclass")
        mlp.fit(X_train, y_one_hot, epochs=50, lr=0.001)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        task: str = "binary",
        random_state: Optional[int] = None,
    ):
        """
        Args:
            layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
            activation: "relu" or "sigmoid" for hidden layers
            task: "binary" (sigmoid out) or "multiclass" (softmax out)
            random_state: For reproducibility
        """
        assert len(layer_sizes) >= 2, "Need at least input and output layers"
        assert activation in ACTIVATIONS, f"activation must be in {list(ACTIVATIONS.keys())}"
        assert task in ("binary", "multiclass"), "task must be 'binary' or 'multiclass'"

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.task = task
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        rng = np.random.default_rng(random_state)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.losses: List[float] = []

        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He init for ReLU, Xavier for Sigmoid
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W = rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X: np.ndarray) -> Tuple[List, List]:
        """Forward pass. Returns (activations_list, pre_activations_list)."""
        activations = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            pre_activations.append(z)

            is_last = i == len(self.weights) - 1
            if is_last:
                if self.task == "binary":
                    a = sigmoid(z)
                else:
                    a = softmax(z)
            else:
                a = self.act_fn(z)
            activations.append(a)

        return activations, pre_activations

    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """Compute loss based on task."""
        eps = 1e-9
        if self.task == "binary":
            # Binary cross-entropy
            return float(-np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)))
        else:
            # Categorical cross-entropy
            return float(-np.mean(np.sum(y * np.log(y_pred + eps), axis=1)))

    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List,
        pre_activations: List,
        lr: float,
        l2: float,
    ) -> None:
        """Backpropagation — update weights in place."""
        n = X.shape[0]
        n_layers = len(self.weights)

        # Output layer gradient (BCE or CE with sigmoid/softmax simplifies to same form)
        delta = activations[-1] - y  # shape: (batch, output)

        for i in range(n_layers - 1, -1, -1):
            # Weight gradient
            dW = (activations[i].T @ delta) / n + l2 * self.weights[i]
            db = np.mean(delta, axis=0, keepdims=True)

            # Propagate delta to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_deriv(pre_activations[i - 1])

            # Update
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.01,
        l2: float = 0.0,
        verbose: bool = True,
    ) -> "MLP":
        """Train the network.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Labels. For binary: (n,) or (n,1). For multiclass: one-hot (n, n_classes)
            epochs: Training epochs
            batch_size: Mini-batch size
            lr: Learning rate
            l2: L2 regularization coefficient
            verbose: Print loss every 10 epochs
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        self.losses = []

        for epoch in range(epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[perm], y[perm]

            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                y_batch = y_shuf[start : start + batch_size]

                activations, pre_acts = self._forward(X_batch)
                epoch_loss += self._compute_loss(activations[-1], y_batch) * len(X_batch)
                self._backward(X_batch, y_batch, activations, pre_acts, lr, l2)

            avg_loss = epoch_loss / n_samples
            self.losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} — Loss: {avg_loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output activation probabilities."""
        activations, _ = self._forward(np.array(X, dtype=float))
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        if self.task == "binary":
            return (proba >= 0.5).astype(int).squeeze()
        else:
            return np.argmax(proba, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy score."""
        y_pred = self.predict(X)
        y_true = np.array(y)
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        return float(np.mean(y_pred == y_true.squeeze()))


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    np.random.seed(42)

    # ── Binary classification ──────────────────────────────────────────
    print("=" * 60)
    print("MLP Binary Classification (synthetic, 2 classes)")
    print("=" * 60)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLP([20, 64, 32, 1], activation="relu", task="binary", random_state=42)
    mlp.fit(X_train, y_train, epochs=100, lr=0.01, batch_size=32, verbose=True)
    print(f"\nOur MLP accuracy:  {mlp.score(X_test, y_test):.4f}")

    sk_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    sk_mlp.fit(X_train, y_train)
    print(f"sklearn MLP accuracy: {sk_mlp.score(X_test, y_test):.4f}")

    # ── Multi-class: Iris ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MLP Multi-class Classification (Iris, 3 classes)")
    print("=" * 60)
    iris = load_iris()
    X2, y2 = iris.data, iris.target
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)

    scaler2 = StandardScaler()
    X_train2 = scaler2.fit_transform(X_train2)
    X_test2 = scaler2.transform(X_test2)

    # One-hot encode
    n_classes = 3
    y_onehot = np.eye(n_classes)[y_train2]

    mlp2 = MLP([4, 32, 16, 3], activation="relu", task="multiclass", random_state=42)
    mlp2.fit(X_train2, y_onehot, epochs=200, lr=0.01, batch_size=16, verbose=False)
    print(f"Our MLP accuracy:  {mlp2.score(X_test2, y_test2):.4f}")

    sk_mlp2 = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    sk_mlp2.fit(X_train2, y_train2)
    print(f"sklearn accuracy:  {sk_mlp2.score(X_test2, y_test2):.4f}")
