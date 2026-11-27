"""P3. Row reduction: RREF with partial pivoting, rank, null space and column space."""
import numpy as np
rng = np.random.default_rng(0)


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


def null_space_basis(A, tol=1e-10):
    """Return a list with one basis vector of null(A) for each free variable."""
    R, pivots = rref(A, tol)
    n = R.shape[1]
    free = [j for j in range(n) if j not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n)
        v[f] = 1.0  # set this free variable to 1, the others to 0
        for row, p in enumerate(pivots):
            v[p] = -R[row, f]  # solve for the pivot variables
        basis.append(v)
    return basis


def col_space_basis(A, tol=1e-10):
    """Return the pivot columns of the original matrix A (a basis of col(A))."""
    A = np.asarray(A, dtype=float)
    _, pivots = rref(A, tol)
    return [A[:, j] for j in pivots]


def main():
    np.set_printoptions(precision=4, suppress=True)

    print("=== Test 1: the 2x3 example ===")
    A = np.array([[1, 2, -1], [3, -1, 4]])
    R, piv = rref(A)
    print(R, "pivots =", piv)
    print("matches expected:", np.allclose(R, [[1, 0, 1], [0, 1, -1]]) and piv == [0, 1])

    print("\n=== Test 2: the matrix F of Example 6.31 ===")
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    R, piv = rref(F)
    print(R, "pivots =", piv)
    print("matches expected:", np.allclose(R, [[1, 0, -1], [0, 1, 2], [0, 0, 0]]))
    print("rank(F) =", rank(F), "  null space basis:", null_space_basis(F))
    print("col space basis:", col_space_basis(F))

    print("\n=== 20 random 5x7 matrices ===")
    ok_null = ok_rn = ok_rank = True
    ranks = set()
    for _ in range(20):
        A = rng.normal(size=(5, 7))
        N = null_space_basis(A)
        ok_null &= all(np.allclose(A @ v, 0) for v in N)
        ok_rn &= rank(A) + len(N) == 7
        ok_rank &= rank(A) == np.linalg.matrix_rank(A)
        ranks.add(rank(A))
    print("A @ v ~ 0 for all v in null_space_basis(A):", ok_null)
    print("rank(A) + nullity(A) == 7 (rank-nullity):  ", ok_rn)
    print("rank(A) == np.linalg.matrix_rank(A):       ", ok_rank)
    print("ranks found:", sorted(ranks), "(the maximum possible is min(5, 7) = 5)")

    print("\n=== A 6x8 matrix of rank 4 ===")
    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    N = null_space_basis(A)
    C = col_space_basis(A)
    print("rank(A) =", rank(A), "  np.linalg.matrix_rank(A) =", np.linalg.matrix_rank(A))
    print("dim null(A) =", len(N), "  dim col(A) =", len(C))
    print("max |A v| over the null space basis:", max(np.abs(A @ v).max() for v in N))
    # the pivot columns of A span col(A): A has the same rank as these 4 columns
    print("rank of the pivot columns of A:", np.linalg.matrix_rank(np.column_stack(C)))


if __name__ == "__main__":
    main()
