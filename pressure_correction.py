# pressure_correction.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

from face_addressed_mesh_2d import Mesh
from sparse_cr import SparseCR, LinearSystem
from fv_scalar import gauss_grad_cell


def compute_face_flux_linear(mesh: Mesh, Uc: np.ndarray, bcU: dict) -> np.ndarray:
    """
    Task 5.2: Fpre = Sf · Uf, with Uf from linear interpolation of Uc.

    Uf:
      - internal face: (U_owner + U_nei)/2
      - boundary face:
          fixedValue -> use prescribed U
          zeroGradient -> use owner U
    """
    nF = len(mesh.faces)
    nC = len(mesh.cells)
    if Uc.shape != (nC, 2):
        raise ValueError("Uc must be (nCells,2)")

    Uf = np.zeros((nF, 2), dtype=float)

    # internal
    for f in range(nF):
        nei = mesh.face_neighbour[f]
        if nei != -1:
            o = mesh.face_owner[f]
            Uf[f, :] = 0.5 * (Uc[o, :] + Uc[nei, :])

    # boundary
    for patch in mesh.patches:
        pname = patch.name
        if pname not in bcU:
            raise ValueError(f"Missing U BC for patch '{pname}'")
        bc = bcU[pname]
        bct = bc.get("type", "zeroGradient")
        for f in patch.face_ids:
            o = mesh.face_owner[f]
            if bct == "fixedValue":
                val = bc.get("value", [0.0, 0.0])
                Uf[f, :] = np.array(val, dtype=float)
            else:
                Uf[f, :] = Uc[o, :]

    Sf = np.array(mesh.Sf, dtype=float)
    F = np.einsum("ij,ij->i", Sf, Uf)
    return F


def divergence_of_face_flux(mesh: Mesh, F: np.ndarray) -> np.ndarray:
    """
    Compute integrated divergence per cell: sum_{faces in cell} (sign * F_face).
    With Sf oriented owner->neigh, F is positive out of owner.
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)
    if F.shape[0] != nF:
        raise ValueError("F length mismatch")

    div = np.zeros(nC, dtype=float)
    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        div[o] += F[f]
        if n != -1:
            div[n] -= F[f]
    return div


def assemble_laplace_variable_gamma(
    mesh: Mesh,
    gamma_cell: np.ndarray,
    bc_phi: dict,
    source: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Assemble div( gamma grad phi ) with cell-centered gamma (orthogonal FV).

    For internal face f between owner o and neighbour n:
        coeff = gamma_f * |Sf| * delta
    where gamma_f is arithmetic mean of gamma_cell(o), gamma_cell(n).

    Boundary faces:
      - fixedValue: Dirichlet (adds to RHS)
      - zeroGradient: Neumann => no contribution
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)
    if gamma_cell.shape[0] != nC:
        raise ValueError("gamma_cell length mismatch")

    trip: List[Tuple[int, int, float]] = []
    b = np.zeros(nC, dtype=float)

    if source is not None:
        if source.shape[0] != nC:
            raise ValueError("source length mismatch")
        b += source

    magSf = np.array(mesh.magSf, dtype=float)
    delta = np.array(mesh.delta, dtype=float)

    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        if n == -1:
            continue
        gamma_f = 0.5 * (gamma_cell[o] + gamma_cell[n])
        coeff = gamma_f * magSf[f] * delta[f]

        # owner row
        trip.append((o, o, +coeff))
        trip.append((o, n, -coeff))
        # neighbour row
        trip.append((n, n, +coeff))
        trip.append((n, o, -coeff))

    # boundary faces
    for patch in mesh.patches:
        pname = patch.name
        bc = bc_phi.get(pname, {"type": "zeroGradient"})
        bct = bc.get("type", "zeroGradient")
        val = float(bc.get("value", 0.0))

        for f in patch.face_ids:
            o = mesh.face_owner[f]
            n = mesh.face_neighbour[f]
            if n != -1:
                continue

            if bct == "fixedValue":
                coeff = gamma_cell[o] * magSf[f] * delta[f]
                trip.append((o, o, +coeff))
                b[o] += coeff * val
            else:
                pass

    return trip, b


def pin_pressure_reference(
    nC: int,
    trip: List[Tuple[int, int, float]],
    b: np.ndarray,
    ref_cell: int = 0,
    ref_value: float = 0.0,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Impose p(ref_cell) = ref_value to remove null-space for all-Neumann pressure.
    Implemented by replacing the row with a unit diagonal.
    """
    if not (0 <= ref_cell < nC):
        raise ValueError("ref_cell out of range")

    trip2 = [(i, j, v) for (i, j, v) in trip if i != ref_cell]
    trip2.append((ref_cell, ref_cell, 1.0))
    b2 = b.copy()
    b2[ref_cell] = ref_value
    return trip2, b2


def _row_coeffs_from_triplets(triplets: List[Tuple[int, int, float]], row: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i, j, v in triplets:
        if i == row:
            out[j] = out.get(j, 0.0) + float(v)
    return out


def solve_pressure_equation(
    mesh: Mesh,
    Fpre: np.ndarray,
    aP_x: np.ndarray,
    aP_y: np.ndarray,
    alphaU: float = 1.0,   # kept for API compatibility; no longer used in body
    bc_p: Optional[dict] = None,
    ref_cell: int = 0,
    gs_sweeps: int = 200,
    p_init: Optional[np.ndarray] = None,
    report_cells: tuple[int, int] | None = None,  # (internal_cell, lid_adjacent_cell)
    return_history: bool = False,
    tol_abs: float = 1e-12,                         
    tol_rel: float = 1e-10,                         
    report_every: int = 2000,                       
    max_sweeps: int = 50000,
) -> np.ndarray | Tuple[np.ndarray, list]:
    """
    Task 5.3: Solve pressure equation (PDF Eq. 5):
        div( (1/aP) grad p ) = div(u)  (FV integrated => RHS = -div(Fpre))

    SIMPLE choice:
        aP = 0.5*(aP_x + aP_y)  (pre-relaxation diagonal from momentum)
        gamma_cell = 1.0/aP      (NOT alphaU/aP — Jasak correction)

    For all-Neumann: pin one reference cell.
    """
    nC = len(mesh.cells)
    if bc_p is None:
        bc_p = {p.name: {"type": "zeroGradient"} for p in mesh.patches}

    if aP_x.shape[0] != nC or aP_y.shape[0] != nC:
        raise ValueError("aP arrays length mismatch")

    aP = 0.5 * (aP_x + aP_y)
    if np.any(aP <= 0.0):
        raise ValueError("Non-positive aP encountered; cannot form 1/aP")

    gamma_cell = 1.0 / aP

    rhs = -divergence_of_face_flux(mesh, Fpre)
    trip, b_lap = assemble_laplace_variable_gamma(mesh, gamma_cell, bc_p, source=None)
    b = b_lap + rhs

    # optional fixedCell reference in bc_p
    ref = bc_p.get("reference", None) if bc_p is not None else None
    if isinstance(ref, dict) and ref.get("type") == "fixedCell":
        ref_cell = int(ref.get("cell", ref_cell))
        ref_value = float(ref.get("value", 0.0))
    else:
        ref_value = 0.0

    trip, b = pin_pressure_reference(nC, trip, b, ref_cell=ref_cell, ref_value=ref_value)

    # Task 9 reporting
    if report_cells is not None:
        c_int, c_lid = report_cells
        for c in [int(c_int), int(c_lid)]:
            if c == ref_cell:
                continue
            row = _row_coeffs_from_triplets(trip, c)
            cols = sorted(row.keys())
            print(
                f"[Task9 Pressure] cell={c} coeffs: "
                + ", ".join([f"({j}:{row[j]:+.6e})" for j in cols])
            )

    A = SparseCR.from_triplets(nC, nC, trip)
    sys = LinearSystem(A, b)
    if p_init is None:
        p0 = np.zeros(nC, dtype=float)
    else:
        if p_init.shape[0] != nC:
            raise ValueError("p_init length mismatch")
        p0 = p_init.copy()

    if return_history:
        p, r0, rf, sweeps, hist = sys.solve_gs_to_tol(
            x0=p0,
            max_sweeps=gs_sweeps,      # <<< use gs_sweeps passed by caller
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            report_every=report_every,
            return_history=True,
            history_every=1,
        )
        print(f"[Pressure] GS sweeps={sweeps}, residual start={r0}, end={rf}, hist_len={len(hist)}")
        return p, hist

    p, r0, rf, sweeps = sys.solve_gs_to_tol(
        x0=p0,
        max_sweeps=gs_sweeps,          # <<< use gs_sweeps passed by caller
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        report_every=report_every,
    )

    print(f"[Pressure] GS sweeps={sweeps}, residual start={r0}, end={rf}")
    return p

def correct_face_flux(
    mesh: Mesh,
    Fpre: np.ndarray,
    p: np.ndarray,
    aP_x: np.ndarray,
    aP_y: np.ndarray,
    alphaU: float = 1.0,   # kept for API compatibility; no longer used in body
) -> np.ndarray:
    """
    Task 5.4 (part 1): Flux correction (PDF Eq. 6):
        Fcorr = Fpre - (1/aP)_f * Sf · (grad p)_f

    For orthogonal meshes:
        Sf · (grad p)_f ≈ |Sf| * delta * (pN - pP)
    """
    nF = len(mesh.faces)
    nC = len(mesh.cells)
    if p.shape[0] != nC:
        raise ValueError("p length mismatch")
    if Fpre.shape[0] != nF:
        raise ValueError("Fpre length mismatch")

    aP = 0.5 * (aP_x + aP_y)
    inv_aP = 1.0 / aP

    magSf = np.array(mesh.magSf, dtype=float)
    delta = np.array(mesh.delta, dtype=float)

    Fcorr = Fpre.copy()

    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        if n == -1:
            continue

        inv_aP_f = 0.5 * (inv_aP[o] + inv_aP[n])
        Sf_dot_gradp = magSf[f] * delta[f] * (p[n] - p[o])
        Fcorr[f] = Fpre[f] - inv_aP_f * Sf_dot_gradp

    return Fcorr


def correct_cell_velocity(
    mesh: Mesh,
    Ustar: np.ndarray,
    p: np.ndarray,
    aP_x: np.ndarray,
    aP_y: np.ndarray,
    alphaU: float = 1.0,   # kept for API compatibility; no longer used in body
    bc_p: Optional[dict] = None,
) -> np.ndarray:
    """
    Task 5.4 (part 2): Cell-centre velocity correction (PDF Eq. 7):
        Ucorr = Ustar - (1/aP) * grad p
    """
    nC = len(mesh.cells)
    if Ustar.shape != (nC, 2):
        raise ValueError("Ustar must be (nCells,2)")
    if p.shape[0] != nC:
        raise ValueError("p length mismatch")

    if bc_p is None:
        bc_p = {patch.name: {"type": "zeroGradient"} for patch in mesh.patches}

    # Clean pressure BC: keep only patch entries (ignore "reference" etc.)
    bc_p_clean = {patch.name: bc_p.get(patch.name, {"type": "zeroGradient"}) for patch in mesh.patches}
    gradp = gauss_grad_cell(mesh, p, bc_p_clean)

    if np.any(aP_x <= 0.0) or np.any(aP_y <= 0.0):
        raise ValueError("Non-positive aP encountered in velocity correction")

    Ucorr = Ustar.copy()
    Ucorr[:, 0] = Ustar[:, 0] - (1.0 / aP_x) * gradp[:, 0]
    Ucorr[:, 1] = Ustar[:, 1] - (1.0 / aP_y) * gradp[:, 1]
    return Ucorr