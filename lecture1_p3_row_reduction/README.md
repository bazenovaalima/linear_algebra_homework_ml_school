# P3. Row reduction (Lecture 1)

Run: `pip install numpy matplotlib`, then `python p3_rref.py`. Full output: [output.txt](output.txt).

`rref(A)` (partial pivoting, tolerance for zero), plus `rank`, `null_space_basis` and `col_space_basis`.

**Results**
- `[[1, 2, -1], [3, -1, 4]]` → `[[1, 0, 1], [0, 1, -1]]`, pivots `[0, 1]`.
- Matrix F (Example 6.31) → `[[1, 0, -1], [0, 1, 2], [0, 0, 0]]`, rank 2, null space spanned by `(1, -2, 1)`.
- 20 random 5×7 matrices: `A v ≈ 0` for the null space basis, rank + nullity = 7, rank agrees with `np.linalg.matrix_rank`.
- 6×8 matrix of rank 4 (product of 6×4 and 4×8): rank 4, null space of dimension 4.
