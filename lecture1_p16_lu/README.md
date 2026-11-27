# P16. LU decomposition and rank-one updates (Lecture 1)

Run: `pip install numpy matplotlib`, then `python p16_lu.py`. Full output: [output.txt](output.txt).

`lu(A)` (elimination with Schur complements, no row swaps), `solve_lu(L, U, b)` (forward and back substitution), `sherman_morrison`.

**Results**
- Example 6.107: `L = [[1,0,0],[2,1,0],[4,3,1]]`, `U = [[2,1,1],[0,1,1],[0,0,2]]`, as in the notes. Random 6×6: `L @ U == A`, and `solve_lu` agrees with `np.linalg.solve`.
- `A = [[1e-17, 1], [1, 1]]`, `b = (1, 2)`: without pivoting the solver returns `(0, 1)` instead of `(1, 1)`. The multiplier is 1e17, so `u22 = 1 − 1e17` rounds to `−1e17` and the entry `A22 = 1` is lost. After swapping the rows (partial pivoting) the result is `(1, 1)`.
- 200×50 `X`, 100 new rows added one at a time with Sherman–Morrison: the result matches `(XᵀX)⁻¹` recomputed from scratch (relative difference 2e-15). One update costs O(d²) instead of O(Nd² + d³) for a recomputation; 100 updates take about 0.6 ms versus about 35 ms for 100 recomputations.
