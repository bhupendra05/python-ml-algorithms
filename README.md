# python-ml-algorithms

Classic machine learning algorithms implemented from scratch using only NumPy, with side-by-side sklearn comparisons. Built for learning, interviews, and understanding the math.

## Algorithms

| Algorithm | File | Type | Key Concepts |
|-----------|------|------|--------------|
| Linear Regression | `supervised/linear_regression.py` | Regression | Gradient descent, MSE, R² |
| Logistic Regression | `supervised/logistic_regression.py` | Classification | Sigmoid, BCE loss, L2 reg |
| K-Nearest Neighbors | `supervised/knn.py` | Both | Lazy learning, Euclidean distance |
| K-Means Clustering | `unsupervised/kmeans.py` | Clustering | K-Means++, inertia, elbow method |
| PCA | `unsupervised/pca.py` | Dimensionality reduction | Eigendecomposition, explained variance |
| Neural Network (MLP) | `neural/mlp.py` | Both | Backprop, ReLU, mini-batch SGD |
| Metrics | `utils/metrics.py` | Evaluation | Accuracy, F1, R², AUC |

## Quick Start

```bash
pip install -r requirements.txt  # numpy + sklearn (sklearn only for comparison/datasets)

# Run any algorithm (each has a demo in __main__)
python supervised/linear_regression.py
python supervised/logistic_regression.py
python supervised/knn.py
python unsupervised/kmeans.py
python unsupervised/pca.py
python neural/mlp.py
python utils/metrics.py
```

## Linear Regression

**Math**: Minimize `MSE = (1/2n) * ||Xw + b - y||²`

**Update rules**:
```
w = w - α * (1/n) * Xᵀ(Xw + b - y)
b = b - α * (1/n) * Σ(Xw + b - y)
```

```python
from supervised.linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))    # R²
print(model.mse(X_test, y_test))      # MSE
print(model.weights, model.bias)
```

Sample output (California Housing):
```
Our R²:    0.5832
sklearn R²: 0.5843
```

## Logistic Regression

**Math**: `P(y=1|x) = σ(wᵀx + b)` where `σ(z) = 1/(1+e^-z)`

**Loss**: Binary Cross-Entropy `L = -(1/n) Σ [y log(p) + (1-y) log(1-p)]`

```python
from supervised.logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.1, epochs=500, l2_lambda=0.01)
model.fit(X_train, y_train)
model.predict(X_test)                  # → [0, 1, 0, 1, ...]
model.predict_proba(X_test)            # → [0.12, 0.87, ...]
model.score(X_test, y_test)            # Accuracy
```

## K-Nearest Neighbors

**How it works**: Store all training data. For each test point, find k nearest neighbors and vote (classification) or average (regression).

```python
from supervised.knn import KNNClassifier, KNNRegressor

# Classification
knn = KNNClassifier(k=5, metric="euclidean")
knn.fit(X_train, y_train)
knn.predict(X_test)
knn.predict_proba(X_test)  # → vote fractions

# Regression
knn_reg = KNNRegressor(k=5, weights="distance")  # inverse-distance weighting
knn_reg.fit(X_train, y_train)
knn_reg.score(X_test, y_test)  # R²
```

## K-Means Clustering

**K-Means++** initialization places centroids far apart, reducing bad convergence:
```
1. Pick first centroid uniformly at random
2. Pick next centroid with probability ∝ distance² to nearest centroid
3. Repeat until k centroids
```

```python
from unsupervised.kmeans import KMeans, elbow_method

km = KMeans(k=3, random_state=42)
km.fit(X)
print(km.labels_)             # [0, 2, 1, 0, ...]
print(km.inertia_)            # 1234.5
print(km.cluster_centers_)   # shape (k, n_features)

# Find best k with elbow method
elbow_method(X, range(1, 10))
```

## PCA

**Math**: Eigendecomposition of covariance matrix

```
C = (1/n-1) XᵀX
C = Q Λ Qᵀ
Z = X_centered @ Q[:, :k]
```

```python
from unsupervised.pca import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)            # (n, 2)
print(pca.explained_variance_ratio_)   # [0.72, 0.23]
print(pca.n_components_for_variance(0.95))  # 5

# Reconstruction
X_approx = pca.inverse_transform(X_2d)
print(pca.reconstruction_error(X))
```

## Neural Network (MLP)

**Architecture**: Input → Dense(ReLU) × L → Output(Sigmoid/Softmax)

**Backpropagation**:
```
δ_output = y_pred - y
δ_hidden = (δ_next @ Wᵀ) * act_derivative(z)
dW = (1/n) * aᵀ @ δ
db = mean(δ)
```

```python
from neural.mlp import MLP

# Binary classification: [input, hidden1, hidden2, output]
mlp = MLP([20, 64, 32, 1], activation="relu", task="binary")
mlp.fit(X_train, y_train, epochs=100, batch_size=32, lr=0.01)
mlp.score(X_test, y_test)   # accuracy

# Multi-class (pass one-hot labels)
mlp = MLP([4, 32, 16, 3], activation="relu", task="multiclass")
y_onehot = np.eye(3)[y_train]
mlp.fit(X_train, y_onehot, epochs=200, lr=0.001)
```

## Metrics

```python
from utils.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mse, rmse, mae, mape, r2_score, huber_loss
)

# Classification
accuracy_score(y_true, y_pred)          # 0.875
precision_score(y_true, y_pred)         # TP/(TP+FP)
recall_score(y_true, y_pred)            # TP/(TP+FN)
f1_score(y_true, y_pred)               # harmonic mean of P and R
roc_auc_score(y_true, y_scores)        # area under ROC curve
print(classification_report(y_true, y_pred))

# Multi-class
f1_score(y_true, y_pred, average="macro")     # unweighted mean
f1_score(y_true, y_pred, average="weighted")  # support-weighted mean

# Regression
mse(y_true, y_pred)    # mean squared error
rmse(y_true, y_pred)   # root MSE (same units as y)
mae(y_true, y_pred)    # mean absolute error (robust to outliers)
r2_score(y_true, y_pred)  # R² ∈ (-∞, 1.0]
```

## Implementation Notes

- **No magic**: every formula is derived in the code comments
- **sklearn for comparison only**: used only in `__main__` blocks and for test datasets
- **Numerical stability**: sigmoid uses conditional exp, softmax shifts by max
- **He/Xavier init**: ReLU layers use `√(2/fan_in)`, Sigmoid uses `√(1/fan_in)`

## Requirements

```
numpy>=1.26.0        # all algorithm implementations
scikit-learn>=1.4.0  # only for comparison in __main__ blocks
```

## License

MIT
