"""P15. Tensors with NumPy: reshaping, unfoldings, einsum, Kronecker products, rank one."""
import numpy as np
rng = np.random.default_rng(0)

import time


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


def rank(A, tol=1e-10):
    """Rank of A = number of pivots of its RREF."""
    return len(rref(A, tol)[1])


def rank_one(u, v, w):
    """Outer product u (x) v (x) w, the tensor with entries u_i v_j w_k."""
    return np.einsum("i,j,k->ijk", u, v, w)


def is_multiple(X, Y, tol=1e-10):
    """True if the matrices X and Y are linearly dependent (one is a multiple of the other)."""
    return rank(np.vstack([X.reshape(-1), Y.reshape(-1)]), tol) <= 1


def timeit(f, repeat=5):
    """Best wall-clock time of f() over a few runs, in seconds."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)

    print("=== Reshaping and unfoldings ===")
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    ok = all(T[i, j, k] == flat[12 * i + 4 * j + k]
             for i in range(2) for j in range(3) for k in range(4))
    print("T[i, j, k] == T.reshape(-1)[12*i + 4*j + k] for all i, j, k:", ok)
    U1 = T.reshape(2, 12)
    U2 = T.transpose(1, 0, 2).reshape(3, 8)
    print("T.reshape(2, 12) =\n", U1)
    print("T.transpose(1, 0, 2).reshape(3, 8) =\n", U2)
    print("ranks:", rank(U1), "and", rank(U2),
          " (np.linalg.matrix_rank:", np.linalg.matrix_rank(U1), np.linalg.matrix_rank(U2), ")")
    U3 = T.transpose(2, 0, 1).reshape(4, 6)
    print("third unfolding T.transpose(2, 0, 1).reshape(4, 6) has rank", rank(U3))

    print("\n=== Table 1 ===")
    A, B = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    S = rng.normal(size=(4, 4))
    x, y = rng.normal(size=4), rng.normal(size=4)
    checks = {
        "matrix product  ij,jk->ik ": np.allclose(np.einsum("ij,jk->ik", A, B), A @ B),
        "trace           ii->      ": np.isclose(np.einsum("ii->", S), np.trace(S)),
        "dot product     i,i->     ": np.isclose(np.einsum("i,i->", x, y), x @ y),
        "outer product   i,j->ij   ": np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y)),
        "transpose       ij->ji    ": np.allclose(np.einsum("ij->ji", A), A.T),
        "bilinear form   i,ij,j->  ": np.isclose(np.einsum("i,ij,j->", x, S, y), x @ S @ y),
    }
    for name, res in checks.items():
        print(name, res)

    print("\n=== A batch of 100 random 4x4 matrices ===")
    Ab, Bb = rng.normal(size=(100, 4, 4)), rng.normal(size=(100, 4, 4))
    tr_einsum = np.einsum("bii->b", Ab)
    prod_einsum = np.einsum("bij,bjk->bik", Ab, Bb)

    def loop():
        tr = np.array([np.trace(Ab[b]) for b in range(100)])
        prods = np.array([Ab[b] @ Bb[b] for b in range(100)])
        return tr, prods

    tr_loop, prod_loop = loop()
    print("traces   einsum('bii->b')        == loop:", np.allclose(tr_einsum, tr_loop))
    print("products einsum('bij,bjk->bik')  == loop:", np.allclose(prod_einsum, prod_loop))
    print("first three traces:", tr_einsum[:3])
    t_e = timeit(lambda: (np.einsum("bii->b", Ab), np.einsum("bij,bjk->bik", Ab, Bb)))
    t_l = timeit(loop)
    print(f"time: einsum {t_e * 1e6:.1f} us, Python loop {t_l * 1e6:.1f} us "
          f"(loop is {t_l / t_e:.0f}x slower)")

    print("\n=== Kronecker product (Remark 11.17) ===")
    A, B, X = rng.normal(size=(3, 4)), rng.normal(size=(2, 5)), rng.normal(size=(4, 5))
    lhs = (A @ X @ B.T).reshape(-1)
    rhs = np.kron(A, B) @ X.reshape(-1)
    print("np.kron(A, B).shape =", np.kron(A, B).shape)
    print("(A @ X @ B.T).reshape(-1) == np.kron(A, B) @ X.reshape(-1):", np.allclose(lhs, rhs))
    n = 100
    print(f"for A, B of size {n}x{n}: np.kron(A, B) is {n * n}x{n * n}, i.e. {n**4:.0e} numbers "
          f"= {n**4 * 8 / 1e9:.1f} GB in float64")
    print(f"multiplications: kron matrix-vector {n**4:.0e}, "
          f"A X B^T directly {2 * n**3:.0e}")
    n = 50  # a size at which the Kronecker product still fits comfortably in memory
    A, B, X = rng.normal(size=(n, n)), rng.normal(size=(n, n)), rng.normal(size=(n, n))
    t_direct = timeit(lambda: A @ X @ B.T)
    t_kron = timeit(lambda: np.kron(A, B) @ X.reshape(-1), repeat=2)
    K = np.kron(A, B)
    t_kron_mv = timeit(lambda: K @ X.reshape(-1), repeat=2)
    same = np.allclose((A @ X @ B.T).reshape(-1), K @ X.reshape(-1))
    print(f"n = {n}: both ways agree: {same}")
    print(f"n = {n}: A @ X @ B.T {t_direct * 1e3:.3f} ms, "
          f"kron + matvec {t_kron * 1e3:.1f} ms (matvec alone {t_kron_mv * 1e3:.1f} ms); "
          f"kron stores {K.size:.2e} numbers")

    print("\n=== Rank-one tensors (Example 11.20) ===")
    u, v, w = rng.normal(size=3), rng.normal(size=4), rng.normal(size=5)
    T = rank_one(u, v, w)
    uvT = np.outer(u, v)
    print("T.shape =", T.shape)
    print("T[:, :, k] == w_k * u v^T for all k:",
          all(np.allclose(T[:, :, k], w[k] * uvT) for k in range(5)))
    print("all slices are multiples of slice 0:",
          all(is_multiple(T[:, :, k], T[:, :, 0]) for k in range(5)))
    # the rank-2 tensor e1(x)e1(x)e1 + e2(x)e2(x)e2 of the example fails this test
    e1, e2 = np.eye(2)
    T2 = rank_one(e1, e1, e1) + rank_one(e2, e2, e2)
    print("e1(x)e1(x)e1 + e2(x)e2(x)e2: slices\n", T2[:, :, 0], "\n", T2[:, :, 1])
    print("are its two slices multiples of each other?", is_multiple(T2[:, :, 0], T2[:, :, 1]))


if __name__ == "__main__":
    main()
