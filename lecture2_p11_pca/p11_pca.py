"""Lecture 2, Programming Exercise P11: principal component analysis.

Run from this folder:  python p11_pca.py
Figures are saved to figures/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)  # random number generator for test data
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)


def pca_eig(X):
    """Principal components of centered data X (N x d) via the covariance matrix.

    Returns the eigenvalues in decreasing order and the eigenvectors as columns.
    """
    N = X.shape[0]
    S = X.T @ X / N
    lam, Q = np.linalg.eigh(S)  # increasing order
    order = np.argsort(lam)[::-1]
    return lam[order], Q[:, order]


def pca_svd(X):
    """Principal components of centered data X via the SVD X = U Sigma V^T.

    Returns sigma_i^2 / N (the eigenvalues of S) and V (the right singular vectors).
    """
    N = X.shape[0]
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    return s**2 / N, Vt.T


def same_up_to_sign(Q1, Q2):
    """Check that the columns of Q1 and Q2 agree up to their signs."""
    signs = np.sign(np.sum(Q1 * Q2, axis=0))
    return np.allclose(Q1, Q2 * signs)


# ---------------------------------------------------------------------------
# Part 1: 2D correlated Gaussian data.
C = np.array([[3.0, 1.5], [1.5, 1.0]])  # covariance matrix
X_raw = rng.multivariate_normal([2, -1], C, size=500)
N = X_raw.shape[0]

mean = X_raw.mean(axis=0)
X = X_raw - mean  # centered data
print("sample mean:", np.round(mean, 4))

lam, Q = pca_eig(X)
lam_svd, V = pca_svd(X)
print("\nPrincipal components via eigh(S) (columns):\n", np.round(Q, 4))
print("Principal components via SVD of X (columns):\n", np.round(V, 4))
print("agree up to sign:", same_up_to_sign(Q, V))
print("eigenvalues of S:      ", np.round(lam, 6))
print("sigma_i^2 / N from SVD:", np.round(lam_svd, 6))
print("lambda_i = sigma_i^2 / N:", np.allclose(lam, lam_svd))

lam_C, Q_C = np.linalg.eigh(C)
order = np.argsort(lam_C)[::-1]
lam_C, Q_C = lam_C[order], Q_C[:, order]
print("\nTrue covariance C: eigenvalues", np.round(lam_C, 4))
print("eigenvectors of C (columns):\n", np.round(Q_C, 4))
cosines = np.abs(np.sum(Q * Q_C, axis=0))
print("angles between sample and true principal directions (degrees):",
      np.round(np.degrees(np.arccos(np.clip(cosines, -1, 1))), 3))

explained = lam[0] / lam.sum()
print(f"\nproportion of variance explained by PC1: {explained:.4f}"
      f"  (for C itself: {lam_C[0] / lam_C.sum():.4f})")

# Plot the data with the principal directions scaled by sqrt(lambda_i).
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(X_raw[:, 0], X_raw[:, 1], s=6, alpha=0.4, label="data")
colors = ["tab:red", "tab:green"]
for i in range(2):
    d = np.sqrt(lam[i]) * Q[:, i]
    ax.annotate("", xy=mean + d, xytext=mean,
                arrowprops=dict(arrowstyle="->", lw=2.5, color=colors[i]))
    ax.plot([], [], color=colors[i], lw=2.5,
            label=rf"PC{i + 1}: $\sqrt{{\lambda_{i + 1}}}\,q_{i + 1}$, $\lambda_{i + 1}$ = {lam[i]:.3f}")
ax.set_aspect("equal")
ax.set_title(f"Data and principal directions (PC1 explains {explained:.1%})")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "p11_principal_directions.png"), dpi=130)
plt.close()

# Projection onto the first principal component.
q1 = Q[:, 0]
z = X @ q1                      # 1D scores
X_proj = np.outer(z, q1)        # reconstructions pi(x_i) = (q1^T x_i) q1 (centered)
recon_error = np.mean(np.sum((X - X_proj) ** 2, axis=1))
print(f"\naverage reconstruction error: {recon_error:.6f}")
print(f"lambda_2:                     {lam[1]:.6f}")
print("equal:", np.isclose(recon_error, lam[1]))
print(f"variance of the scores z = {z.var():.6f} = lambda_1 = {lam[0]:.6f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.3, label="centered data")
ax.scatter(X_proj[:, 0], X_proj[:, 1], s=6, color="tab:red", label="projection onto PC1")
for i in range(0, N, 25):  # a few residual segments
    ax.plot([X[i, 0], X_proj[i, 0]], [X[i, 1], X_proj[i, 1]], color="gray", lw=0.7)
ax.set_aspect("equal")
ax.set_title("Projection onto the first principal component")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "p11_projection.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# Part 2: 500 points in R^10 close to a 2-dimensional subspace.
X10 = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 10))
X10 = X10 + 0.05 * rng.normal(size=(500, 10))  # small noise
X10 = X10 - X10.mean(axis=0)
lam10, _ = pca_eig(X10)
print("\n10D data: eigenvalues of the covariance matrix:")
print(np.array2string(lam10, precision=5))
print(f"proportion explained by the first two PCs: {lam10[:2].sum() / lam10.sum():.6f}")
print(f"noise variance 0.05^2 = {0.05**2:.5f}; mean of the last 8 eigenvalues = {lam10[2:].mean():.5f}")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.semilogy(np.arange(1, 11), lam10, "o-")
ax.axhline(0.05**2, color="gray", ls="--", lw=1, label=r"noise variance $0.05^2$")
ax.set_xticks(range(1, 11))
ax.set_xlabel("index $i$")
ax.set_ylabel(r"$\lambda_i$ (log scale)")
ax.set_title("Eigenvalues of the covariance matrix (10D data)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "p11_scree_10d.png"), dpi=130)
plt.close()
