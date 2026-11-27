"""P12. Dual bases and polynomial interpolation (Lagrange polynomials, Runge's phenomenon)."""
import numpy as np
rng = np.random.default_rng(0)

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rref(A, tol=1e-10):
    """Return (R, pivots): the RREF of A and the list of pivot columns."""
    R = np.array(A, dtype=float)  # work on a copy of A
    m, n = R.shape
    pivots = []
    r = 0  # row in which the next pivot will be placed
    for c in range(n):
        if r == m:
            break
        # partial pivoting: largest entry in column c among rows r, r+1, ...
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < tol:
            R[r:, c] = 0.0  # no pivot: clean up the rounding noise in this column
            continue
        R[[r, p]] = R[[p, r]]  # swap rows r and p
        R[r] = R[r] / R[r, c]  # make the pivot equal to 1
        for i in range(m):  # clear all other entries of column c
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def inverse(A, tol=1e-12):
    """Inverse of A by the Gauss-Jordan method: row reduce (A | I) to (I | A^{-1})."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R, _ = rref(np.hstack([A, np.eye(n)]), tol)
    if not np.allclose(R[:, :n], np.eye(n)):
        raise np.linalg.LinAlgError("matrix is singular")
    return R[:, n:]


def dual_basis(P):
    """Return the rows r_1, ..., r_n of the dual basis of the columns of P.

    By Proposition 8.7 these are the rows of P^{-1}:
    r_i b_j = (P^{-1} P)_{ij} = delta_ij.  Row i of the result is r_i.
    """
    return inverse(P)


def vandermonde(t, d):
    """Matrix with entries V[k, j] = t_k ** j, j = 0..d (rows = evaluation functionals)."""
    t = np.asarray(t, dtype=float)
    V = np.ones((len(t), d + 1))
    for j in range(1, d + 1):
        V[:, j] = V[:, j - 1] * t
    return V


def lagrange_basis(nodes):
    """Coefficient vectors of the Lagrange polynomials L_0..L_d as columns of V^{-1}."""
    nodes = np.asarray(nodes, dtype=float)
    return inverse(vandermonde(nodes, len(nodes) - 1))


def interpolate(f, nodes):
    """Coefficients of p = sum_k f(t_k) L_k, the interpolating polynomial of f."""
    C = lagrange_basis(nodes)  # column k = coefficients of L_k
    return C @ f(np.asarray(nodes, dtype=float))


def poly_eval(coeffs, x):
    """Evaluate sum_j coeffs[j] x^j by Horner's scheme."""
    y = np.zeros_like(x, dtype=float)
    for c in coeffs[::-1]:
        y = y * x + c
    return y


def lebesgue_constant(nodes, xs):
    """max over xs of sum_k |L_k(t)|: how much interpolation can amplify errors in f(t_k)."""
    C = lagrange_basis(nodes)
    values = np.array([poly_eval(C[:, k], xs) for k in range(len(nodes))])
    return np.abs(values).sum(axis=0).max()


def main():
    np.set_printoptions(precision=4, suppress=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Dual basis of the columns of a random 5x5 matrix P ===")
    P = rng.normal(size=(5, 5))
    Rows = dual_basis(P)
    G = np.array([[Rows[i] @ P[:, j] for j in range(5)] for i in range(5)])
    print("max |r_i b_j - delta_ij| =", np.abs(G - np.eye(5)).max())
    x = rng.normal(size=5)
    coords = Rows @ x  # r_i x = i-th coordinate of x in the basis b_1..b_5
    x_rebuilt = sum(coords[i] * P[:, i] for i in range(5))
    print("x = sum_i (r_i x) b_i:", np.allclose(x, x_rebuilt),
          "  max error", np.abs(x - x_rebuilt).max())
    print("dual_basis(P) close to np.linalg.inv(P):", np.allclose(Rows, np.linalg.inv(P)))

    print("\n=== Lagrange polynomials for the nodes -1, 0, 1 ===")
    nodes = np.array([-1.0, 0.0, 1.0])
    V = vandermonde(nodes, 2)
    print("V =\n", V)
    C = lagrange_basis(nodes)
    print("V^{-1} (column k = coefficients of L_k in the basis 1, x, x^2) =\n", C)
    expected = np.array([[0, 1, 0], [-0.5, 0, 0.5], [0.5, -1, 0.5]])  # x(x-1)/2, 1-x^2, x(x+1)/2
    print("equals the coefficients of x(x-1)/2, 1-x^2, x(x+1)/2:", np.allclose(C, expected))
    print("L_j(t_k) = (V C)_{kj} = delta_jk:", np.allclose(V @ C, np.eye(3)))
    print("values 1, 2, 5 at -1, 0, 1 give the coefficients", C @ np.array([1.0, 2.0, 5.0]),
          "(expected 2 + 2x + x^2)")

    print("\n=== Interpolation ===")
    cases = [
        ("sin(3t), 9 nodes on [-1, 1]", lambda t: np.sin(3 * t), -1.0, 1.0, 9, "p12_sin.png"),
        ("1/(1+t^2), 21 nodes on [-5, 5]", lambda t: 1 / (1 + t**2), -5.0, 5.0, 21,
         "p12_runge.png"),
    ]
    for name, f, a, b, n, fname in cases:
        nodes = np.linspace(a, b, n)
        coeffs = interpolate(f, nodes)
        # check against NumPy's solver for V c = f(t)
        coeffs_np = np.linalg.solve(vandermonde(nodes, n - 1), f(nodes))
        xs = np.linspace(a, b, 2001)
        err = np.abs(poly_eval(coeffs, xs) - f(xs))
        print(f"{name}: max |p(t_k) - f(t_k)| = "
              f"{np.abs(poly_eval(coeffs, nodes) - f(nodes)).max():.2e}, "
              f"max |p - f| on the interval = {err.max():.3e}, "
              f"agrees with np.linalg.solve: {np.allclose(coeffs, coeffs_np, atol=1e-6)}")
        middle = np.abs(xs) <= (b - a) / 10
        print(f"    max error on the middle fifth = {err[middle].max():.3e}, "
              f"max error near the ends = {err[~middle].max():.3e} "
              f"(at t = {xs[np.argmax(err)]:.3f})")

        runge = n == 21
        fig, axes = plt.subplots(1, 2 if runge else 1, figsize=(11 if runge else 6, 3.6),
                                 squeeze=False)
        for ax in axes[0]:
            ax.plot(xs, f(xs), "k", lw=2, label="f")
            ax.plot(xs, poly_eval(coeffs, xs), "C1", lw=1.5,
                    label=f"p, {n} equally spaced nodes")
            ax.plot(nodes, f(nodes), "o", color="C1", ms=4)
            ax.set_xlabel("t")
        axes[0, 0].set_title(name)
        if runge:
            # for comparison: Chebyshev nodes, clustered near the ends
            cheb = (a + b) / 2 + (b - a) / 2 * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
            c_cheb = interpolate(f, cheb)
            err_cheb = np.abs(poly_eval(c_cheb, xs) - f(xs)).max()
            print(f"    with {n} Chebyshev nodes instead: max |p - f| = {err_cheb:.3e}")
            print(f"    Lebesgue constant max_t sum_k |L_k(t)|: equally spaced "
                  f"{lebesgue_constant(nodes, xs):.3e}, Chebyshev "
                  f"{lebesgue_constant(cheb, xs):.3f}")
            axes[0, 1].plot(xs, poly_eval(c_cheb, xs), "C2--", lw=1.2,
                            label=f"p, {n} Chebyshev nodes")
            axes[0, 1].set_ylim(-1.0, 2.0)
            axes[0, 1].set_title("zoom, with Chebyshev nodes for comparison")
        for ax in axes[0]:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"    saved figures/{fname}")


if __name__ == "__main__":
    main()
