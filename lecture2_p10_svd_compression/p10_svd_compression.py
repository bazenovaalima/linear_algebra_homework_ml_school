"""Lecture 2, Programming Exercise P10: image compression with the SVD.

We use two grayscale images:
  * "synthetic": sin(x) cos(y) + x/10 on a 200 x 300 grid plus a little noise
    (the image suggested in the exercise);
  * "photo": the sample photo grace_hopper.jpg shipped with Matplotlib,
    converted to grayscale by averaging the color channels.

Run from this folder:  python p10_svd_compression.py
Figures are saved to figures/.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)  # random number generator for test data
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)


def synthetic_image(m=200, n=300, noise=0.05):
    """sin(x) cos(y) + x/10 on an m x n grid, plus Gaussian noise."""
    x = np.linspace(0, 10, m)[:, None]  # rows
    y = np.linspace(0, 10, n)[None, :]  # columns
    return np.sin(x) * np.cos(y) + x / 10 + noise * rng.normal(size=(m, n))


def photo_image():
    """The sample photo of Matplotlib as a grayscale array with values in [0, 1]."""
    path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    return plt.imread(path).mean(axis=2) / 255.0


def rank_k_approximation(U, s, Vt, k):
    """A_k = sum_{i<=k} sigma_i u_i v_i^T (Eckart--Young)."""
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def analyze(name, A, ks=(1, 5, 20, 50)):
    m, n = A.shape
    print(f"\n=== {name}: {m} x {n} image ===")
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Decay of the singular values.
    print("sigma_1, sigma_2, sigma_5, sigma_20, sigma_50, sigma_last:",
          np.round([s[0], s[1], s[4], s[19], s[49], s[-1]], 3))
    energy = np.cumsum(s**2) / np.sum(s**2)
    for frac in (0.9, 0.99, 0.999):
        print(f"  rank needed for {frac:.1%} of ||A||_F^2: {np.searchsorted(energy, frac) + 1}")

    plt.figure(figsize=(5, 3.2))
    plt.semilogy(np.arange(1, len(s) + 1), s, ".-", ms=3)
    plt.xlabel("index $i$")
    plt.ylabel(r"$\sigma_i$ (log scale)")
    plt.title(f"Singular values: {name}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"p10_singular_values_{name}.png"), dpi=130)
    plt.close()

    # Rank-k approximations.
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3 * (len(ks) + 1), 3 * m / n + 0.6))
    axes[0].imshow(A, cmap="gray")
    axes[0].set_title("original")
    print(f"  {'k':>3} {'||A - A_k||_F':>14} {'sqrt(sum s_i^2, i>k)':>21} {'rel. error':>10} {'storage':>8}")
    for ax, k in zip(axes[1:], ks):
        Ak = rank_k_approximation(U, s, Vt, k)
        err = np.linalg.norm(A - Ak)  # Frobenius norm
        tail = np.sqrt(np.sum(s[k:] ** 2))
        ratio = k * (m + n + 1) / (m * n)
        assert np.isclose(err, tail), (err, tail)
        print(f"  {k:>3} {err:>14.6f} {tail:>21.6f} {err / np.linalg.norm(A):>10.4f} {ratio:>8.2%}")
        ax.imshow(Ak, cmap="gray")
        ax.set_title(f"$k = {k}$ ({ratio:.1%} storage)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"p10_approximations_{name}.png"), dpi=130)
    plt.close()

    # ||A||_2 = sigma_1 and ||A||_F^2 = sum sigma_i^2.
    print(f"  ||A||_2   = {np.linalg.norm(A, 2):.10f},  sigma_1       = {s[0]:.10f}")
    print(f"  ||A||_F^2 = {np.linalg.norm(A, 'fro')**2:.10f},  sum sigma_i^2 = {np.sum(s**2):.10f}")
    assert np.isclose(np.linalg.norm(A, 2), s[0])
    assert np.isclose(np.linalg.norm(A, "fro") ** 2, np.sum(s**2))


if __name__ == "__main__":
    analyze("synthetic", synthetic_image())
    analyze("photo", photo_image())
    print("\nAll checks passed.")
