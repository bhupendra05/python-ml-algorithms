"""K-Means Clustering from scratch with K-Means++ initialization.

Algorithm:
1. Initialize k centroids using K-Means++ (better than random)
2. Assignment step: assign each point to nearest centroid
3. Update step: move centroid to mean of assigned points
4. Repeat until convergence (centroids stop moving)

Complexity: O(n * k * d * iterations)
"""
import numpy as np
from typing import Optional


class KMeans:
    """K-Means clustering with K-Means++ centroid initialization.

    K-Means++ picks initial centroids with probability proportional to
    squared distance from existing centroids — prevents bad initialization.

    Attributes:
        cluster_centers_: Centroid coordinates, shape (k, n_features)
        labels_: Cluster assignment per sample, shape (n_samples,)
        inertia_: Sum of squared distances to nearest centroid
        n_iter_: Number of iterations until convergence

    Example:
        km = KMeans(k=3, random_state=42)
        km.fit(X)
        print(km.labels_)
        print(km.inertia_)
    """

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init: str = "kmeans++",
    ):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def _init_centroids_random(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random centroid initialization."""
        indices = rng.choice(len(X), self.k, replace=False)
        return X[indices].copy()

    def _init_centroids_plusplus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ initialization — spreads centroids across data.

        1. Pick first centroid uniformly at random.
        2. For each subsequent centroid, pick with probability proportional
           to squared distance from nearest existing centroid.
        """
        n_samples = len(X)
        centroids = []

        # Pick first centroid randomly
        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.k):
            # Squared distance to nearest centroid for each point
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            # Sample proportional to squared distance
            probs = dists / dists.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to nearest centroid (Euclidean)."""
        # Shape: (n_samples, k)
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(dists, axis=1)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Sum of squared distances to assigned centroids."""
        total = 0.0
        for k in range(self.k):
            mask = labels == k
            if mask.any():
                total += np.sum((X[mask] - centroids[k]) ** 2)
        return total

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means to data.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)
        X = np.array(X, dtype=float)

        # Initialize centroids
        if self.init == "kmeans++":
            centroids = self._init_centroids_plusplus(X, rng)
        else:
            centroids = self._init_centroids_random(X, rng)

        labels = None
        for i in range(self.max_iter):
            # E-step: assign labels
            new_labels = self._assign_labels(X, centroids)

            # M-step: update centroids
            new_centroids = np.array([
                X[new_labels == k].mean(axis=0) if (new_labels == k).any() else centroids[k]
                for k in range(self.k)
            ])

            # Check convergence (centroid shift)
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            labels = new_labels
            self.n_iter_ = i + 1

            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new data points to nearest cluster."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._assign_labels(np.array(X), self.cluster_centers_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_


def elbow_method(X: np.ndarray, k_range: range, random_state: int = 42):
    """Compute inertia for different k values (for elbow plot).

    Returns:
        List of (k, inertia) tuples
    """
    results = []
    for k in k_range:
        km = KMeans(k=k, random_state=random_state)
        km.fit(X)
        results.append((k, km.inertia_))
        print(f"  k={k}: inertia={km.inertia_:.2f}, iters={km.n_iter_}")
    return results


if __name__ == "__main__":
    from sklearn.datasets import make_blobs, load_iris
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.metrics import adjusted_rand_score

    np.random.seed(42)

    print("=" * 60)
    print("K-Means on synthetic blobs (3 clusters, 300 samples)")
    print("=" * 60)
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

    our_km = KMeans(k=3, random_state=42)
    our_km.fit(X)
    print(f"Our K-Means:")
    print(f"  Inertia:    {our_km.inertia_:.2f}")
    print(f"  Iterations: {our_km.n_iter_}")
    print(f"  ARI score:  {adjusted_rand_score(y_true, our_km.labels_):.4f}")

    sk_km = SklearnKMeans(n_clusters=3, random_state=42, n_init=1, init="k-means++")
    sk_km.fit(X)
    print(f"\nsklearn K-Means:")
    print(f"  Inertia:    {sk_km.inertia_:.2f}")
    print(f"  Iterations: {sk_km.n_iter_}")
    print(f"  ARI score:  {adjusted_rand_score(y_true, sk_km.labels_):.4f}")

    print("\n" + "=" * 60)
    print("Elbow method (finding optimal k)")
    print("=" * 60)
    elbow_method(X, range(1, 8))

    print("\n" + "=" * 60)
    print("K-Means on Iris dataset (3 real clusters)")
    print("=" * 60)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    km_iris = KMeans(k=3, random_state=42)
    km_iris.fit(X_iris)
    ari = adjusted_rand_score(y_iris, km_iris.labels_)
    print(f"ARI (higher = better cluster-label alignment): {ari:.4f}")
    print(f"Inertia: {km_iris.inertia_:.2f}")
