# P15. Tensors with NumPy (Lecture 1)

Run: `pip install numpy matplotlib`, then `python p15_tensors.py`. Full output: [output.txt](output.txt).

**Results**
- `T = np.arange(24).reshape(2, 3, 4)`: `T[i, j, k] == T.reshape(-1)[12i + 4j + k]` for all indices. Both unfoldings `T.reshape(2, 12)` and `T.transpose(1, 0, 2).reshape(3, 8)` have rank 2, since `T_ijk = 12i + 4j + k`.
- All operations of the einsum table (`ij,jk->ik`, `ii->`, `i,i->`, `i,j->ij`, `ij->ji`, `i,ij,j->`) agree with `@`, `np.trace`, `np.outer`, `.T`.
- Batch of 100 matrices 4×4: traces `bii->b` and products `bij,bjk->bik` in one call each, equal to a Python loop (about 10× faster).
- `(A X Bᵀ).reshape(-1) == np.kron(A, B) @ X.reshape(-1)`. For 100×100 `A, B`, `np.kron(A, B)` stores 10⁸ numbers (0.8 GB) and needs 10⁸ multiplications, while `A X Bᵀ` needs only 2·10⁶.
- `rank_one(u, v, w)`: every slice `T[:, :, k]` equals `w_k · u vᵀ`.
