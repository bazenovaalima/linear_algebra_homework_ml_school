"""P6. Visualizing linear transformations of the plane, and the shoelace formula."""
import numpy as np
rng = np.random.default_rng(0)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Shapes are stored with one point per row; the image of P under A is P @ A.T.
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
LETTER_F = np.array([[0.2, 0.1], [0.4, 0.1], [0.4, 0.45], [0.65, 0.45], [0.65, 0.6],
                     [0.4, 0.6], [0.4, 0.75], [0.8, 0.75], [0.8, 0.9], [0.2, 0.9]])


def grid_lines(k=4):
    """Segments x = t and y = t (t = 1/k, ..., (k-1)/k) inside the unit square."""
    ts = np.arange(1, k) / k
    vertical = [np.array([[t, 0], [t, 1]]) for t in ts]
    horizontal = [np.array([[0, t], [1, t]]) for t in ts]
    return vertical + horizontal


def plot_transformation(A, ax=None, title=None):
    """Draw the unit square, a grid and the letter F, before and after A."""
    A = np.asarray(A, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    # before: dashed square and gray F
    ax.add_patch(Polygon(SQUARE, closed=True, fill=False, ec="gray", ls="--", lw=1))
    ax.add_patch(Polygon(LETTER_F, closed=True, fc="0.8", ec="none"))
    # after: blue parallelogram with the image of the grid, orange F
    ax.add_patch(Polygon(SQUARE @ A.T, closed=True, fc=(0.2, 0.4, 0.9, 0.15),
                         ec="navy", lw=1.5))
    for seg in grid_lines():
        img = seg @ A.T
        ax.plot(img[:, 0], img[:, 1], color="cornflowerblue", lw=0.8)
    ax.add_patch(Polygon(LETTER_F @ A.T, closed=True, fc="darkorange", ec="none"))
    # images of the standard basis vectors (the columns of A)
    for col, color in [(A[:, 0], "firebrick"), (A[:, 1], "forestgreen")]:
        ax.annotate("", xy=col, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
    # axes and limits that contain both shapes
    pts = np.vstack([SQUARE, SQUARE @ A.T])
    lo, hi = pts.min(axis=0) - 0.4, pts.max(axis=0) + 0.4
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def shoelace_area(P):
    """Signed area of the polygon with corners P[0], ..., P[k-1] (shoelace formula)."""
    x, y = P[:, 0], P[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)  # (x_{i+1}, y_{i+1}), cyclically
    return 0.5 * np.sum(x * y_next - x_next * y)


MATRICES = {
    "(a) stretch": np.array([[2, 0], [0, 1]]),
    "(b) rotation by 90 degrees": np.array([[0, -1], [1, 0]]),
    "(c) reflection in the x-axis": np.array([[1, 0], [0, -1]]),
    "(d) shear": np.array([[1, 1], [0, 1]]),
    "(e) singular": np.array([[1, 2], [2, 4]]),
    "(f) matrix of Figure 4": np.array([[3, 1], [1, 2]]),
}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Gallery: the four panels of Figure 2 plus the singular matrix
    # and the matrix of Figure 4.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, A) in zip(axes.flat, MATRICES.items()):
        plot_transformation(A, ax, f"{name}\nA = {A.tolist()}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p6_gallery.png"), dpi=150)
    plt.close(fig)
    print("saved figures/p6_gallery.png")

    print("\n=== Shoelace formula on the image of the unit square ===")
    print(f"{'matrix':30s} {'shoelace area':>14s} {'det(A)':>8s}  equal?")
    for name, A in MATRICES.items():
        area = shoelace_area(SQUARE @ A.T)
        d = np.linalg.det(A)
        print(f"{name:30s} {area:14.4f} {d:8.4f}  {np.isclose(area, d)}")

    # sanity checks of the shoelace formula itself
    print("\nunit square counterclockwise:", shoelace_area(SQUARE),
          "  clockwise:", shoelace_area(SQUARE[::-1]))
    print("area of the letter F:", shoelace_area(LETTER_F))
    ok = True
    for _ in range(100):
        A = rng.normal(size=(2, 2))
        ok &= np.isclose(shoelace_area(SQUARE @ A.T), np.linalg.det(A))
        # areas of all shapes are scaled by det(A), e.g. the letter F
        ok &= np.isclose(shoelace_area(LETTER_F @ A.T),
                         np.linalg.det(A) * shoelace_area(LETTER_F))
    print("100 random 2x2 matrices: shoelace area of A(square) == det(A) and "
          "area(A(F)) == det(A) * area(F):", ok)


if __name__ == "__main__":
    main()
