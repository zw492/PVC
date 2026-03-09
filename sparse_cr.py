# Task 4
# sparse_cr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class SparseCR:
    """
    Compressed Row sparse matrix.
    Data stored row-wise:
      row_ptr: size (nrows+1)
      col_ind: size nnz
      val:     size nnz
    """
    nrows: int
    ncols: int
    row_ptr: np.ndarray
    col_ind: np.ndarray
    val: np.ndarray

    @staticmethod
    def from_triplets(nrows: int, ncols: int, triplets: List[Tuple[int, int, float]]) -> "SparseCR":
        """
        Build CR matrix from (i,j,aij) entries.
        Duplicates are summed.
        """
        if nrows <= 0 or ncols <= 0:
            raise ValueError("nrows and ncols must be positive")

        # Sort by (row, col)
        triplets_sorted = sorted(triplets, key=lambda x: (x[0], x[1]))

        # Combine duplicates
        combined: List[Tuple[int, int, float]] = []
        for i, j, v in triplets_sorted:
            if i < 0 or i >= nrows or j < 0 or j >= ncols:
                raise IndexError(f"Triplet index out of bounds: ({i},{j})")
            if not combined or (combined[-1][0] != i or combined[-1][1] != j):
                combined.append((i, j, float(v)))
            else:
                ii, jj, vv = combined[-1]
                combined[-1] = (ii, jj, vv + float(v))

        # Count nnz per row
        nnz_per_row = np.zeros(nrows, dtype=int)
        for i, _, _ in combined:
            nnz_per_row[i] += 1

        row_ptr = np.zeros(nrows + 1, dtype=int)
        row_ptr[1:] = np.cumsum(nnz_per_row)

        nnz = row_ptr[-1]
        col_ind = np.empty(nnz, dtype=int)
        val = np.empty(nnz, dtype=float)

        # Fill
        cursor = row_ptr[:-1].copy()
        for i, j, v in combined:
            k = cursor[i]
            col_ind[k] = j
            val[k] = v
            cursor[i] += 1

        return SparseCR(nrows=nrows, ncols=ncols, row_ptr=row_ptr, col_ind=col_ind, val=val)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.ncols:
            raise ValueError("Dimension mismatch in matvec")
        y = np.zeros(self.nrows, dtype=float)
        for i in range(self.nrows):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            y[i] = np.dot(self.val[start:end], x[self.col_ind[start:end]])
        return y

    def get_diag(self) -> np.ndarray:
        d = np.zeros(self.nrows, dtype=float)
        for i in range(self.nrows):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            cols = self.col_ind[start:end]
            vals = self.val[start:end]
            mask = (cols == i)
            if np.any(mask):
                d[i] = vals[mask][0]
        return d


@dataclass
class LinearSystem:
    A: "SparseCR"
    b: np.ndarray

    def residual(self, x: np.ndarray) -> np.ndarray:
        return self.b - self.A.matvec(x)

    def residual_norm2(self, x: np.ndarray) -> float:
        r = self.residual(x)
        return float(np.sqrt(np.dot(r, r)))

    def gauss_seidel(self, x: np.ndarray, nsweeps: int = 1) -> np.ndarray:
        """
        Do nsweeps Gauss–Seidel sweeps for Ax=b (in-place update on x).
        """
        n = self.A.nrows
        if x.shape[0] != n:
            raise ValueError("x has wrong size")

        for _ in range(nsweeps):
            for i in range(n):
                start, end = self.A.row_ptr[i], self.A.row_ptr[i + 1]
                cols = self.A.col_ind[start:end]
                vals = self.A.val[start:end]

                a_ii = None
                sigma = 0.0
                for c, aic in zip(cols, vals):
                    if c == i:
                        a_ii = aic
                    else:
                        sigma += aic * x[c]

                if a_ii is None or abs(a_ii) < 1e-30:
                    raise ZeroDivisionError(f"Missing/zero diagonal at row {i}")

                x[i] = (self.b[i] - sigma) / a_ii

        return x

    def solve_gs_to_tol(
        self,
        x0: np.ndarray | None = None,
        max_sweeps: int = 20000,
        tol_abs: float = 1e-8,
        tol_rel: float = 1e-10,
        report_every: int = 200,
        return_history: bool = False,
        history_every: int = 1,
    ):
        """
        Iterate GS until residual <= max(tol_abs, tol_rel * r0) or max_sweeps reached.
        Default return:
            (x, r0, r_final, sweeps_used)
        If return_history=True, returns:
            (x, r0, r_final, sweeps_used, history)
        where history is a list of (sweep_index, residual_norm2).
        """
        n = self.A.nrows
        x = np.zeros(n, dtype=float) if x0 is None else np.array(x0, dtype=float, copy=True)
        if x.shape[0] != n:
            raise ValueError("x0 has wrong size")
        if history_every < 1:
            raise ValueError("history_every must be >= 1")

        r0 = self.residual_norm2(x)
        target = max(tol_abs, tol_rel * r0)

        history = []
        if return_history:
            history.append((0, float(r0)))

        # Already converged?
        if r0 <= target:
            if return_history:
                return x, r0, r0, 0, history
            return x, r0, r0, 0

        for k in range(1, max_sweeps + 1):
            self.gauss_seidel(x, nsweeps=1)

            if return_history and (k % history_every) == 0:
                rk_now = float(self.residual_norm2(x))
                history.append((k, rk_now))

            if (k % report_every) == 0 or k == 1:
                rk = self.residual_norm2(x)
                if rk <= target:
                    if return_history:
                        if not history or history[-1][0] != k:
                            history.append((k, float(rk)))
                        return x, r0, rk, k, history
                    return x, r0, rk, k

        rf = self.residual_norm2(x)
        if return_history:
            if not history or history[-1][0] != max_sweeps:
                history.append((max_sweeps, float(rf)))
            return x, r0, rf, max_sweeps, history

        return x, r0, rf, max_sweeps
