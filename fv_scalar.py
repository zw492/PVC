# fv_scalar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np

from face_addressed_mesh_2d import Mesh

Vec2 = Tuple[float, float]


def interp_face_scalar(mesh: Mesh, phi: np.ndarray) -> np.ndarray:
    """
    Linear interpolation to faces using mesh.fx:
      phi_f = fx*phi_P + (1-fx)*phi_N
    Boundary faces default to phi_f = phi_P (fx=1); BC handler may override.
    """
    nF = len(mesh.faces)
    phi_f = np.zeros(nF, dtype=float)
    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        if n != -1:
            fx = mesh.fx[f]
            phi_f[f] = fx * phi[o] + (1.0 - fx) * phi[n]
        else:
            phi_f[f] = phi[o]
    return phi_f


def gauss_grad_cell(mesh: Mesh, phi: np.ndarray, bc: Dict[str, Any] | None = None) -> np.ndarray:
    """
    Cell-centre gradient using Gauss theorem:
      grad(phi)_P = (1/V_P) * sum_f (phi_f * Sf_f)
    In 2D, V_P is area.

    bc (optional): boundary conditions for phi.
      bc[patchName] = {"type":"fixedValue","value":...} or {"type":"zeroGradient"}
    If fixedValue, we override phi_f on those boundary faces.
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)
    grad = np.zeros((nC, 2), dtype=float)

    phi_f = interp_face_scalar(mesh, phi)

    # Apply fixedValue BC to face values if provided
    if bc is not None:
        patch_faces = {p.name: p.face_ids for p in mesh.patches}
        # B1: require BC specification for every patch in the mesh
        missing_patches = [pname for pname in patch_faces.keys() if pname not in bc]
        if missing_patches:
            raise ValueError(f"Missing BC specification for patches: {missing_patches}")

        for pname, spec in bc.items():
            if pname not in patch_faces:
                raise ValueError(f"BC specified for unknown patch '{pname}'. Known patches: {list(patch_faces.keys())}")

            if spec.get("type") == "fixedValue":
                val = float(spec["value"])
                for f in patch_faces[pname]:
                    phi_f[f] = val
            # zeroGradient: leave phi_f = phi_P (already the default for boundary faces)


    # accumulate face contributions into owner and neighbour with correct orientation
    # mesh.Sf is oriented owner->neighbour; for neighbour cell, flux uses -Sf
    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        sf = mesh.Sf[f]
        pf = phi_f[f]

        grad[o, 0] += pf * sf[0]
        grad[o, 1] += pf * sf[1]
        if n != -1:
            grad[n, 0] -= pf * sf[0]
            grad[n, 1] -= pf * sf[1]

    # divide by cell area
    for c in range(nC):
        A = mesh.cell_areas[c]
        if A <= 0:
            raise ValueError(f"Cell {c} has non-positive area")
        grad[c, :] /= A

    return grad


def assemble_laplace(
    mesh: Mesh,
    gamma: float,
    bc: Dict[str, Any],
    source: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """
    Assemble diffusion operator for:
      -div(gamma * grad(phi)) = source

    Returns:
      triplets: list of (row, col, value) for SparseCR
      b: RHS vector

    Discretisation (standard FV):
      For internal face f between P(owner) and N(neighbour):
        a_PN += gamma * |Sf| * delta
        Adds to matrix:
          A[P,P] += a
          A[P,N] -= a
          A[N,N] += a
          A[N,P] -= a

      For boundary face with Dirichlet (fixedValue):
        treat as neighbour at boundary:
          a = gamma * |Sf| * delta_b
          A[P,P] += a
          b[P] += a * phi_B

      For boundary face with zeroGradient:
        no contribution (Neumann 0 flux)

    bc format:
      bc[patchName] = {"type":"fixedValue","value":float} OR {"type":"zeroGradient"}
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)

    if source is None:
        b = np.zeros(nC, dtype=float)
    else:
        b = np.array(source, dtype=float).copy()
        if b.shape[0] != nC:
            raise ValueError("source has wrong length")

    triplets: list[tuple[int, int, float]] = []

    # Map patch name -> face ids for quick access
    patch_faces = {p.name: p.face_ids for p in mesh.patches}
    missing_patches = [pname for pname in patch_faces.keys() if pname not in bc]
    if missing_patches:
        raise ValueError(f"Missing BC specification for patches: {missing_patches}")

    # Build a face -> BC spec lookup for boundary faces (strict)
    face_bc: Dict[int, Dict[str, Any]] = {}
    for pname, spec in bc.items():
        if pname not in patch_faces:
            raise ValueError(
                f"BC specified for unknown patch '{pname}'. Known patches: {list(patch_faces.keys())}"
            )
        for f in patch_faces[pname]:
            face_bc[f] = spec

    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]

        a = gamma * mesh.magSf[f] * mesh.delta[f]

        if n != -1:
            # internal coupling
            triplets.append((o, o, +a))
            triplets.append((o, n, -a))
            triplets.append((n, n, +a))
            triplets.append((n, o, -a))
        else:
            if f not in face_bc:
                raise ValueError(f"No BC assigned to boundary face {f}. Check boundary patches/BC dict.")
            spec = face_bc[f]

            bctype = spec.get("type", "zeroGradient")

            if bctype == "fixedValue":
                phiB = float(spec["value"])
                triplets.append((o, o, +a))
                b[o] += a * phiB
            elif bctype == "zeroGradient":
                # zero flux => no contribution
                pass
            else:
                raise ValueError(f"Unknown BC type '{bctype}' on face {f}. Spec={spec}")

    return triplets, b


def check_laplace_symmetry(triplets: List[Tuple[int,int,float]], n: int, tol: float = 1e-12) -> None:
    """
    Debug helper: checks A_ij == A_ji for i!=j (symmetry) for Laplace diffusion assembly.
    Uses a dict of off-diagonal entries.
    """
    A = {}
    for i, j, v in triplets:
        if i == j:
            continue
        A[(i, j)] = A.get((i, j), 0.0) + v

    # Check symmetry on off-diagonals
    for (i, j), v in list(A.items()):
        vji = A.get((j, i), 0.0)
        if abs(v - vji) > tol:
            raise ValueError(f"Matrix not symmetric: A[{i},{j}]={v} vs A[{j},{i}]={vji}")
        

def compute_face_flux_const_u(mesh: Mesh, u: Tuple[float, float]) -> np.ndarray:
    """
    Face flux F = Sf · u, using mesh.Sf oriented owner->neighbour (outward for owner).
    Returns F array of length nFaces.
    """
    ux, uy = float(u[0]), float(u[1])
    nF = len(mesh.faces)
    F = np.zeros(nF, dtype=float)
    for f in range(nF):
        sf = mesh.Sf[f]
        F[f] = sf[0] * ux + sf[1] * uy
    return F


def assemble_convection_upwind(
    mesh: Mesh,
    u: Tuple[float, float],
    bc: Dict[str, Any],
    source: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """
    Assemble steady convection term:
        div(u * phi) = source
    using first-order upwind on faces.

    LHS discretisation for each face uses a single upwind value:
      internal face (owner o, neighbour n) with F = u·Sf (Sf owner->neigh):
        if F >= 0: phi_f = phi_o
          contributes:  row o: +F*phi_o   => A[o,o] += +F
                        row n: -F*phi_o  => A[n,o] += -F
        if F < 0:  phi_f = phi_n
          contributes:  row o: +F*phi_n   => A[o,n] += +F   (note F negative)
                        row n: -F*phi_n  => A[n,n] += -F   (note -F positive)

      boundary face (neigh=-1), owner=o, with outward Sf for owner:
        if F > 0 (outflow): phi_f = phi_o  => A[o,o] += F
        if F < 0 (inflow):  phi_f = phi_B if fixedValue
             move known term to RHS: b[o] += -F * phi_B
          (If inflow with zeroGradient, that’s not well-posed; we error.)

    bc format:
      bc[patchName] = {"type":"fixedValue","value":float} OR {"type":"zeroGradient"}
    """
    nC = len(mesh.cells)
    nF = len(mesh.faces)

    if source is None:
        b = np.zeros(nC, dtype=float)
    else:
        b = np.array(source, dtype=float).copy()
        if b.shape[0] != nC:
            raise ValueError("source has wrong length")

    triplets: list[tuple[int, int, float]] = []

    # Patch -> face ids
    patch_faces = {p.name: p.face_ids for p in mesh.patches}
    missing_patches = [pname for pname in patch_faces.keys() if pname not in bc]
    if missing_patches:
        raise ValueError(f"Missing BC specification for patches: {missing_patches}")

    # Face -> BC spec for boundary faces
    face_bc: Dict[int, Dict[str, Any]] = {}
    for pname, spec in bc.items():
        if pname not in patch_faces:
            raise ValueError(f"BC specified for unknown patch '{pname}'. Known: {list(patch_faces.keys())}")
        for f in patch_faces[pname]:
            face_bc[f] = spec

    F = compute_face_flux_const_u(mesh, u)

    for f in range(nF):
        o = mesh.face_owner[f]
        n = mesh.face_neighbour[f]
        flux = F[f]

        if n != -1:
            if flux >= 0.0:
                # owner uses phi_o
                triplets.append((o, o, +flux))
                triplets.append((n, o, -flux))
            else:
                # owner uses phi_n (flux < 0)
                triplets.append((o, n, +flux))   # flux is negative
                triplets.append((n, n, -flux))   # -flux is positive
        else:
            # boundary
            if flux > 0.0:
                # outflow: phi_f = phi_o
                triplets.append((o, o, +flux))
            elif flux < 0.0:
                # inflow: need fixedValue
                if f not in face_bc:
                    raise ValueError(f"No BC assigned to boundary face {f}.")
                spec = face_bc[f]
                bctype = spec.get("type", "zeroGradient")
                if bctype == "fixedValue":
                    phiB = float(spec["value"])
                    b[o] += (-flux) * phiB  # since flux < 0, -flux > 0
                else:
                    raise ValueError(
                        f"Inflow boundary face {f} has BC '{bctype}'. "
                        f"Upwind convection needs fixedValue on inflow."
                    )
            else:
                # flux == 0: no contribution
                pass

    return triplets, b

def assemble_ddt(
    mesh: Mesh,
    dt: float,
    phi_old: np.ndarray,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """
    Backward Euler time term for scalar:
        (phi - phi_old)/dt  -> contributes to Ax=b as
        A[P,P] += Vp/dt
        b[P]   += Vp*phi_old/dt

    Returns (triplets, b) ; can add it to other operators.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    nC = len(mesh.cells)
    phi_old = np.asarray(phi_old, dtype=float)
    if phi_old.shape[0] != nC:
        raise ValueError("phi_old has wrong length")

    triplets: list[tuple[int, int, float]] = []
    b = np.zeros(nC, dtype=float)

    for P in range(nC):
        Vp = mesh.cell_areas[P]  # 2-D control-volume area
        aP = Vp / dt
        triplets.append((P, P, aP))
        b[P] += aP * phi_old[P]

    return triplets, b