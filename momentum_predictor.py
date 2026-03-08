# momentum_predictor.py
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

from face_addressed_mesh_2d import Mesh
from sparse_cr import SparseCR, LinearSystem


def interp_face_vector(mesh: Mesh, Uc: np.ndarray, bcU: dict) -> np.ndarray:
    """
    Linear interpolation of cell-centre velocity to faces.
    Returns Uf of shape (nF, 2).
      - internal faces: average owner/neigh
      - boundary faces:
          fixedValue -> use prescribed value
          zeroGradient -> use owner value
    """
    nF = len(mesh.faces)
    Uf = np.zeros((nF, 2), dtype=float)

    # internal faces
    for f in range(nF):
        nei = mesh.face_neighbour[f]
        if nei != -1:
            o = mesh.face_owner[f]
            Uf[f, :] = 0.5 * (Uc[o, :] + Uc[nei, :])

    # boundary faces
    for patch in mesh.patches:
        pname = patch.name
        if pname not in bcU:
            raise ValueError(f"Missing velocity BC for patch '{pname}'")
        bc = bcU[pname]
        bct = bc.get("type", "zeroGradient")
        for f in patch.face_ids:
            o = mesh.face_owner[f]
            if bct == "fixedValue":
                val = bc.get("value", [0.0, 0.0])
                Uf[f, :] = np.array(val, dtype=float)
            else:
                Uf[f, :] = Uc[o, :]

    return Uf


def compute_face_flux(mesh: Mesh, Uc: np.ndarray, bcU: dict) -> np.ndarray:
    """
    Compute Fpre = Sf · Uf for every face. Returns F of shape (nF,).
    """
    Uf = interp_face_vector(mesh, Uc, bcU)
    Sf = np.array(mesh.Sf, dtype=float)
    return np.einsum("ij,ij->i", Sf, Uf)


def assemble_convection_upwind_from_flux(
    mesh: Mesh,
    F: np.ndarray,
    bc_phi: dict,
    source: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Assemble div(F * phi) with upwind using provided face flux F (= Sf·Uf).

    Discrete convention:
      - internal faces contribute via upwind split based on sign(F)
      - boundary faces:
          fixedValue required on inflow (F<0 into owner)
          zeroGradient allowed on outflow
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)
    if F.shape[0] != nF:
        raise ValueError("F length mismatch")

    trip: List[Tuple[int, int, float]] = []
    b = np.zeros(nC, dtype=float)

    if source is not None:
        if source.shape[0] != nC:
            raise ValueError("source length mismatch")
        b += source

    # internal faces
    for f in range(nF):
        nei = mesh.face_neighbour[f]
        if nei == -1:
            continue
        o = mesh.face_owner[f]
        Ff = float(F[f])

        if Ff >= 0.0:
            # upwind = owner
            trip.append((o, o, +Ff))
            trip.append((nei, o, -Ff))
        else:
            # upwind = neighbour (Ff is negative)
            trip.append((o, nei, +Ff))
            trip.append((nei, nei, -Ff))

    # boundary faces
    for patch in mesh.patches:
        pname = patch.name
        bc = bc_phi.get(pname, None)
        if bc is None:
            raise ValueError(f"Missing BC for patch '{pname}'")
        bct = bc.get("type", "zeroGradient")
        val = float(bc.get("value", 0.0))

        for f in patch.face_ids:
            o = mesh.face_owner[f]
            Ff = float(F[f])

            if bct == "fixedValue":
                # inflow uses boundary value as upwind
                if Ff < 0.0:
                    b[o] -= Ff * val
                else:
                    trip.append((o, o, +Ff))
            else:
                # zeroGradient: only allowed on outflow
                if Ff < 0.0:
                    raise ValueError(
                        f"Inflow boundary face {f} on patch '{pname}' has zeroGradient; "
                        "upwind convection needs fixedValue on inflow."
                    )
                if Ff > 0.0:
                    trip.append((o, o, +Ff))

    return trip, b


def assemble_laplace_scalar(
    mesh: Mesh,
    gamma: float,
    bc_phi: dict,
    source: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    """
    Laplacian assembly wrapper (delegates to fv_scalar.assemble_laplace).
    """
    from fv_scalar import assemble_laplace
    return assemble_laplace(mesh, gamma=gamma, bc=bc_phi, source=source)


def apply_implicit_under_relaxation(
    nC: int,
    triplets: List[Tuple[int, int, float]],
    b: np.ndarray,
    phi_old: np.ndarray,
    alpha: float,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray, np.ndarray]:
    """
    Implicit under-relaxation:
      A phi = b  ->  (A/alpha) phi = b + (1-alpha)/alpha * diag(A) * phi_old

    Returns:
      trip_relaxed, b_relaxed, aP (diag(A) of original A)
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")

    aP = np.zeros(nC, dtype=float)
    for (i, j, v) in triplets:
        if i == j:
            aP[i] += float(v)

    trip_relaxed = [(i, j, (v / alpha) if (i == j) else v) for (i, j, v) in triplets]

    b_relaxed = b.copy()
    b_relaxed += ((1.0 - alpha) / alpha) * aP * phi_old

    return trip_relaxed, b_relaxed, aP


def _row_coeffs_from_triplets(triplets: List[Tuple[int, int, float]], row: int) -> Dict[int, float]:
    """
    Build a {col: value} dict for a given row from a triplet list.
    If duplicate (row,col) appears, values are summed.
    """
    out: Dict[int, float] = {}
    for i, j, v in triplets:
        if i == row:
            out[j] = out.get(j, 0.0) + float(v)
    return out


def momentum_predictor(
    mesh: Mesh,
    U: np.ndarray,
    nu: float,
    bcU: dict,
    alphaU: float = 0.7,
    gs_sweeps: int = 200,
    report_cells: tuple[int, int] | None = None,  # (internal_cell, boundary_adjacent_cell)
    return_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """
    Task 5.1 momentum predictor (pressure gradient omitted):
      div(U U) - div(nu grad U) = 0

    Solves two scalar transport problems:
      Ux* and Uy*

    Returns:
      - default: (Ustar (nC,2), aP_x (nC,), aP_y (nC,))
      - if return_history: adds hist_all dict with keys 'Ux','Uy' containing GS residual histories
    """
    nC = len(mesh.cells)
    if U.shape != (nC, 2):
        raise ValueError("U shape must be (nCells,2)")

    # Face flux from current velocity (linearised convection)
    Fpre = compute_face_flux(mesh, U, bcU)

    Ustar = U.copy()
    aP_all: List[np.ndarray] = []
    hist_all: Dict[str, list] = {}

    for comp, comp_name in [(0, "Ux"), (1, "Uy")]:
        # scalar BC dict for this component
        bc_phi = {}
        for patch in mesh.patches:
            pname = patch.name
            bc = bcU.get(pname, None)
            if bc is None:
                raise ValueError(f"Missing U BC for patch '{pname}'")

            if bc.get("type") == "fixedValue":
                v = bc.get("value", [0.0, 0.0])
                bc_phi[pname] = {"type": "fixedValue", "value": float(v[comp])}
            else:
                bc_phi[pname] = {"type": "zeroGradient"}

        # Assemble convection + diffusion
        trip_c, b_c = assemble_convection_upwind_from_flux(mesh, Fpre, bc_phi, source=None)
        trip_d, b_d = assemble_laplace_scalar(mesh, gamma=nu, bc_phi=bc_phi, source=None)

        trip = trip_c + trip_d
        b = b_c + b_d  # RHS is zero for momentum predictor

        # Under-relaxation (implicit)
        phi_old = U[:, comp].copy()
        trip_rel, b_rel, aP = apply_implicit_under_relaxation(nC, trip, b, phi_old, alphaU)

        # Optional Task 8 reporting: matrix coefficients for selected cells
        if report_cells is not None:
            c_int, c_bnd = report_cells
            for c in [int(c_int), int(c_bnd)]:
                row = _row_coeffs_from_triplets(trip_rel, c)
                cols = sorted(row.keys())
                print(
                    f"[Task8 Momentum {comp_name}] cell={c} coeffs: "
                    + ", ".join([f"({j}:{row[j]:+.6e})" for j in cols])
                )

        # Solve implicitly with fixed GS sweeps (disable early stopping)
        A = SparseCR.from_triplets(nC, nC, trip_rel)
        sys = LinearSystem(A, b_rel)
        phi = phi_old.copy()

        if return_history:
            phi, r0, rf, sweeps, hist = sys.solve_gs_to_tol(
                x0=phi,
                max_sweeps=gs_sweeps,
                tol_abs=-1.0,
                tol_rel=-1.0,
                report_every=gs_sweeps,
                return_history=True,
                history_every=1,
            )
            hist_all[comp_name] = hist
            print(
                f"[Momentum predictor {comp_name}] GS sweeps={sweeps}, "
                f"residual start={r0}, end={rf}, hist_len={len(hist)}"
            )
        else:
            phi, r0, rf, sweeps = sys.solve_gs_to_tol(
                x0=phi,
                max_sweeps=gs_sweeps,
                tol_abs=-1.0,
                tol_rel=-1.0,
                report_every=gs_sweeps,
            )
            print(f"[Momentum predictor {comp_name}] GS sweeps={sweeps}, residual start={r0}, end={rf}")

        Ustar[:, comp] = phi
        aP_all.append(aP)

    if return_history:
        return Ustar, aP_all[0], aP_all[1], hist_all
    return Ustar, aP_all[0], aP_all[1]