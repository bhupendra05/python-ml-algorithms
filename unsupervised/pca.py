"""Principal Component Analysis (PCA) from scratch.

PCA finds directions (principal components) that maximize variance.

Algorithm:
1. Center the data: X = X - mean(X)
2. Compute covariance matrix: C = (1/(n-1)) * X^T X
3. Eigendecomposition: C = Q Λ Q^T
4. Sort eigenvectors by descending eigenvalues
5. Project: Z = X @ Q[:, :k]

The math:
- Eigenvectors of covariance matrix are the principal components
- Eigenvalues tell us how much variance each component captures
- explained_variance_ratio_[i] = λ_i / sum(λ)
"""
import numpy as np
from typing import Optional


class PCA:
    """Principal Component Analysis via eigendecomposition of covariance matrix.

    Attributes:
        components_: Principal component vectors, shape (n_components, n_features)
        explained_variance_: Variance captured by each component
        explained_variance_ratio_: Fraction of total variance per component
        mean_: Feature means (subtracted during transform)

    Example:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        print(pca.explained_variance_ratio_)
        # [0.72, 0.23] → 95% variance in 2 components
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.n_samples_: int = 0

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA: compute principal components.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            self
        """
        X = np.array(X, dtype=float)
        self.n_samples_, self.n_features_ = X.shape

        # Determine number of components
        n_components = self.n_components or self.n_features_
        n_components = min(n_components, self.n_features_, self.n_samples_)

        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Covariance matrix
        # Shape: (n_features, n_features)
        cov_matrix = np.cov(X_centered.T)  # (n-1) normalization

        # Step 3: Eigendecomposition
        # np.linalg.eigh returns eigenvalues in ascending order for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort by descending eigenvalue
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Step 5: Select top-k components
        # components_ shape: (n_components, n_features)
        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]

        total_variance = np.sum(np.maximum(eigenvalues, 0))
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(n_components)

        self.singular_values_ = np.sqrt(
            np.maximum(self.explained_variance_, 0) * (self.n_samples_ - 1)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto principal components.

        Args:
            X: Data matrix, shape (n_samples, n_features)

        Returns:
            Reduced data, shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() first.")
        X_centered = np.array(X) - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct approximate original data from reduced representation.

        Args:
            X_reduced: Reduced data, shape (n_samples, n_components)

        Returns:
            Reconstructed data, shape (n_samples, n_features)
        """
        return X_reduced @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error (measures information loss)."""
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        return float(np.mean((X - X_reconstructed) ** 2))

    def n_components_for_variance(self, variance_ratio: float = 0.95) -> int:
        """How many components needed to explain given variance fraction?

        Args:
            variance_ratio: Target explained variance (e.g., 0.95 for 95%)

        Returns:
            Minimum number of components needed
        """
        cumulative = np.cumsum(self.explained_variance_ratio_)
        n = np.searchsorted(cumulative, variance_ratio) + 1
        return int(min(n, len(self.explained_variance_ratio_)))


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)

    print("=" * 60)
    print("PCA on Iris dataset (4D → 2D)")
    print("=" * 60)
    iris = load_iris()
    X = iris.data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Our PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)

    print(f"Our PCA:")
    print(f"  Input shape:  {X_scaled.shape}")
    print(f"  Output shape: {X_reduced.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Total explained: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"  Reconstruction error: {pca.reconstruction_error(X_scaled):.6f}")

    # sklearn comparison
    sk_pca = SklearnPCA(n_components=2)
    X_sk = sk_pca.fit_transform(X_scaled)
    print(f"\nsklearn PCA:")
    print(f"  Explained variance ratio: {sk_pca.explained_variance_ratio_}")
    print(f"  Total explained: {sk_pca.explained_variance_ratio_.sum():.4f}")

    print("\n" + "=" * 60)
    print("PCA on Digits (64D) — finding optimal n_components")
    print("=" * 60)
    digits = load_digits()
    X_digits = digits.data.astype(float)

    full_pca = PCA()  # all components
    full_pca.fit(X_digits)

    for target in [0.80, 0.90, 0.95, 0.99]:
        n = full_pca.n_components_for_variance(target)
        print(f"  {target*100:.0f}% variance → {n} components (of {X_digits.shape[1]})")

    print("\nVariance by component:")
    for i, (ev, evr) in enumerate(
        zip(full_pca.explained_variance_[:10], full_pca.explained_variance_ratio_[:10])
    ):
        print(f"  PC{i+1:2d}: var={ev:.2f}, ratio={evr:.4f}")
