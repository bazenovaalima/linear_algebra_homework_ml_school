"""P16. The LU decomposition and rank-one updates (Sherman-Morrison)."""
import numpy as np
rng = np.random.default_rng(0)

import time


def lu(A):
    """Return L, U with A = L @ U (no row swaps), L unit lower and U upper triangular.

    Elimination as in the proof of Theorem 6.106: write A = [[a, r^T], [c, A']],
    store the multipliers c / a in L, put (a, r^T) into U and continue with the
    Schur complement S = A' - c r^T / a.
    """
    S = np.array(A, dtype=float)  # S holds the current Schur complement in S[k:, k:]
    n = S.shape[0]
    L, U = np.eye(n), np.zeros((n, n))
    for k in range(n):
        a = S[k, k]
        if a == 0:
            raise ZeroDivisionError(f"zero pivot in step {k}; row swaps are needed")
        U[k, k:] = S[k, k:]  # pivot row (a, r^T)
        L[k + 1:, k] = S[k + 1:, k] / a  # multipliers c / a
        # Schur complement A' - (c / a) r^T
        S[k + 1:, k + 1:] -= np.outer(L[k + 1:, k], S[k, k + 1:])
    return L, U


def plu(A):
    """LU with partial pivoting: return P, L, U with P @ A = L @ U (Theorem 6.106 (2))."""
    S = np.array(A, dtype=float)
    n = S.shape[0]
    perm = np.arange(n)
    L, U = np.eye(n), np.zeros((n, n))
    for k in range(n):
        p = k + np.argmax(np.abs(S[k:, k]))  # largest pivot candidate in column k
        S[[k, p]] = S[[p, k]]  # swap rows of the remaining matrix ...
        L[[k, p], :k] = L[[p, k], :k]  # ... and the multipliers stored so far
        perm[[k, p]] = perm[[p, k]]
        U[k, k:] = S[k, k:]
        L[k + 1:, k] = S[k + 1:, k] / S[k, k]
        S[k + 1:, k + 1:] -= np.outer(L[k + 1:, k], S[k, k + 1:])
    return np.eye(n)[perm], L, U


def solve_lu(L, U, b):
    """Solve L U x = b by forward and back substitution.

    b may be a vector or an n x k matrix (then all k systems are solved at once).
    """
    n = L.shape[0]
    y = np.array(b, dtype=float)
    for i in range(n):  # forward substitution: L y = b (L has ones on the diagonal)
        y[i] = (y[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = y
    for i in range(n - 1, -1, -1):  # back substitution: U x = y
        x[i] = (x[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse_lu(A):
    """Inverse of A by solving A X = I with the LU decomposition."""
    L, U = lu(A)
    return solve_lu(L, U, np.eye(A.shape[0]))


def sherman_morrison(Ainv, u, v):
    """Return (A + u v^T)^{-1} from A^{-1} (Theorem 6.108 with k = 1)."""
    Au = Ainv @ u
    vA = v @ Ainv
    return Ainv - np.outer(Au, vA) / (1.0 + v @ Au)


def best_time(f, repeat=5):
    """Best wall-clock time of f() over a few runs, in seconds."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    np.set_printoptions(precision=4, suppress=False, linewidth=100)

    print("=== Example 6.107 ===")
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    L, U = lu(A)
    print("L =\n", L, "\nU =\n", U)
    print("L, U as in the notes:",
          np.allclose(L, [[1, 0, 0], [2, 1, 0], [4, 3, 1]]) and
          np.allclose(U, [[2, 1, 1], [0, 1, 1], [0, 0, 2]]))
    print("det(A) = prod diag(U) =", np.prod(np.diag(U)))
    print("solution of A x = (5, 13, 31):", solve_lu(L, U, np.array([5, 13, 31])))

    print("\n=== Random 6x6 matrices ===")
    ok_lu = ok_solve = True
    worst = 0.0
    for _ in range(100):
        A = rng.normal(size=(6, 6))
        L, U = lu(A)
        ok_lu &= np.allclose(L @ U, A) and np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))
        b = rng.normal(size=6)
        x = solve_lu(L, U, b)
        x_np = np.linalg.solve(A, b)
        ok_solve &= np.allclose(x, x_np)
        worst = max(worst, np.abs(x - x_np).max())
    print("100 matrices: L @ U == A, L lower and U upper triangular:", ok_lu)
    print("solve_lu agrees with np.linalg.solve:", ok_solve,
          f"(largest difference {worst:.1e})")

    print("\n=== A tiny pivot ===")
    A = np.array([[1e-17, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    L, U = lu(A)
    print("L =\n", L, "\nU =\n", U)
    print("L @ U =\n", L @ U, "  (should be A)")
    print("solve_lu without pivoting:", solve_lu(L, U, b))
    P, L, U = plu(A)
    print("with partial pivoting (rows swapped):", solve_lu(L, U, P @ b))
    print("np.linalg.solve:", np.linalg.solve(A, b))
    print("exact solution: x1 = 1/(1 - 1e-17), x2 = (1 - 2e-17)/(1 - 1e-17), both 1 to 16 digits")

    print("\n=== Partial pivoting on random matrices ===")
    ok = True
    for _ in range(20):
        A = rng.normal(size=(6, 6))
        P, L, U = plu(A)
        ok &= np.allclose(P @ A, L @ U) and np.abs(L).max() <= 1 + 1e-12
    print("P @ A == L @ U and all multipliers |L_ij| <= 1:", ok)

    print("\n=== Sherman-Morrison updates of (X^T X)^{-1} ===")
    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))
    M = X.T @ X
    Minv = inverse_lu(M)  # X^T X is positive definite, so no row swaps are needed
    print("initial inverse: max |M Minv - I| =", np.abs(M @ Minv - np.eye(50)).max())

    # approach 1: add the rows one at a time; X^T X changes by the rank-one x x^T
    def updates():
        Minv_sm = Minv.copy()
        for x in new_rows:
            Minv_sm = sherman_morrison(Minv_sm, x, x)
        return Minv_sm

    # approach 2: recompute from scratch for the final 300 x 50 matrix
    X_final = np.vstack([X, new_rows])

    def from_scratch():
        return inverse_lu(X_final.T @ X_final)

    # approach 2': recompute from scratch after each new row (same output as approach 1)
    def from_scratch_each_row(inv=inverse_lu):
        for m in range(1, 101):
            Xm = np.vstack([X, new_rows[:m]])
            Minv_m = inv(Xm.T @ Xm)
        return Minv_m

    Minv_sm, Minv_new = updates(), from_scratch()
    t_sm, t_one = best_time(updates), best_time(from_scratch)
    t_all = best_time(from_scratch_each_row)
    t_np = best_time(lambda: from_scratch_each_row(np.linalg.inv))
    Minv_np = np.linalg.inv(X_final.T @ X_final)
    diff = np.abs(Minv_sm - Minv_new).max()
    print(f"max |SM result - recomputed inverse| = {diff:.2e} "
          f"(relative {diff / np.abs(Minv_new).max():.2e})")
    print("both agree with np.linalg.inv:",
          np.allclose(Minv_sm, Minv_np), np.allclose(Minv_new, Minv_np))
    print(f"time, 100 Sherman-Morrison updates:                  {t_sm * 1e3:8.2f} ms "
          f"({t_sm / 100 * 1e6:.0f} us per update)")
    print(f"time, one inverse from scratch (own LU):             {t_one * 1e3:8.2f} ms "
          f"(incl. forming X^T X)")
    print(f"time, 100 inverses from scratch, one per row (own LU): {t_all * 1e3:6.2f} ms "
          f"-> {t_all / t_sm:.0f}x slower than Sherman-Morrison")
    print(f"time, 100 inverses from scratch with np.linalg.inv:  {t_np * 1e3:8.2f} ms")


if __name__ == "__main__":
    main()
