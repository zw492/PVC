# simple_solver.py
from __future__ import annotations
import json
import numpy as np
from typing import Tuple

from face_addressed_mesh_2d import Mesh
from momentum_predictor import momentum_predictor
from pressure_correction import (
    compute_face_flux_linear,
    solve_pressure_equation,
    correct_face_flux,
    correct_cell_velocity,
    divergence_of_face_flux,
)


def simple_solve(
    case_dir: str,
    n_outer: int = 50,
    alphaU: float = 0.7,
    gs_sweeps: int = 200,
    tol_div_inf: float = 1e-8,
    tol_dU_inf: float = 1e-10,
    ref_cell: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Task 5.5: Outside SIMPLE loop correcting the linearised momentum convection term.

    Loop structure:
      1) Momentum predictor for U* using convection linearised with current flux Fpre
         (in our implementation, momentum_predictor builds convection using Fpre computed
          from current U each call)
      2) Compute Fpre from interpolated face velocity U*
      3) Solve pressure equation for pressure correction p'
      4) Correct face flux and cell velocity using p'
      5) Under-relax pressure: p += p_prime
      6) Repeat until convergence

    Notes:
      - Pressure equation uses Neumann BCs by default (zeroGradient), so we pin a reference cell.
      - Only pressure gradients matter; pinning does not affect U.
    """
    mesh = Mesh.from_folder(f"{case_dir}/mesh")
    bc = json.loads(open(f"{case_dir}/bc.json").read())
    params = json.loads(open(f"{case_dir}/params.json").read())
    nu = float(params["nu"])

    U = np.loadtxt(f"{case_dir}/0/U.txt")
    p = np.loadtxt(f"{case_dir}/0/p.txt")

    bcU = bc["U"]
    bcp = bc.get("p", None)
    if bcp is not None:
        bcp = {k: v for k, v in bcp.items() if k in {patch.name for patch in mesh.patches}}

    if verbose:
        print(f"[SIMPLE] case='{case_dir}', nCells={len(mesh.cells)}, nu={nu}")
        print(f"[SIMPLE] alphaU={alphaU}, GS sweeps/call={gs_sweeps}")

    p_prime_init = np.zeros(len(mesh.cells))   # warm-start buffer for pressure

    for it in range(1, n_outer + 1):
        if verbose and (it == 1 or it % 20 == 0):
            print(f"[Progress] {it}/{n_outer}")
        U_old = U.copy()
        p_old = p.copy()

        # ---- 5.1 Momentum predictor (no pressure gradient)
        Ustar, aP_x, aP_y = momentum_predictor(
            mesh, U, nu, bcU, alphaU=alphaU, gs_sweeps=gs_sweeps
        )

        # ---- 5.2 Precursor flux from interpolated face velocity (Eq 4)
        Fpre = compute_face_flux_linear(mesh, Ustar, bcU)
        boundary_faces = [f for f in range(len(mesh.faces)) if mesh.face_neighbour[f] == -1]
        print("[check] sum(Fpre) on boundary faces =", float(np.sum(Fpre[boundary_faces])))

        # sanity: impermeable walls => boundary flux should be ~0
        boundary_faces = [f for f in range(len(mesh.faces)) if mesh.face_neighbour[f] == -1]
        max_bnd_flux = float(np.max(np.abs(Fpre[boundary_faces])))
        print(f"[check] max |Fpre| on boundary faces = {max_bnd_flux:.3e}")

        # ---- 5.3 Pressure equation (Eq 5): solve for pressure correction p'
        p_prime = solve_pressure_equation(
            mesh, Fpre, aP_x, aP_y, alphaU=alphaU, bc_p=bcp, ref_cell=ref_cell, gs_sweeps=gs_sweeps,
            p_init=p_prime_init            # warm-start: reuse previous p_prime
        )
        p_prime_init = p_prime.copy()      # update for next iteration

        # ---- 5.4 Flux + velocity correction using p'
        Fcorr = correct_face_flux(mesh, Fpre, p_prime, aP_x, aP_y, alphaU=alphaU)
        Ucorr = correct_cell_velocity(mesh, Ustar, p_prime, aP_x, aP_y, alphaU=alphaU, bc_p=bcp)
        div_pre = divergence_of_face_flux(mesh, Fpre)
        div_corr = divergence_of_face_flux(mesh, Fcorr)
        print("[check] ||div(Fpre)||_inf =", float(np.max(np.abs(div_pre))))
        print("[check] ||div(Fcorr)||_inf =", float(np.max(np.abs(div_corr))))
        print("[check] sum(Fcorr) on boundary faces =", float(np.sum(Fcorr[boundary_faces])))

        # Under-relax pressure accumulation (common SIMPLE practice)
        p += p_prime
        U = Ucorr

        # Convergence measures (continuity + velocity change)
        divF = divergence_of_face_flux(mesh, Fcorr)
        div_inf = float(np.max(np.abs(divF)))
        dU_inf = float(np.max(np.abs(U - U_old)))

        if verbose:
            print(
                f"[SIMPLE iter {it:03d}] "
                f"||div(F)||_inf={div_inf:.3e}  "
                f"||dU||_inf={dU_inf:.3e}"
            )

        if div_inf < tol_div_inf and dU_inf < tol_dU_inf:
            if verbose:
                print(f"[SIMPLE] Converged at iter {it}: div_inf={div_inf:.3e}, dU_inf={dU_inf:.3e}")
            break

    # Save final fields
    np.savetxt(f"{case_dir}/U_final.txt", U)
    np.savetxt(f"{case_dir}/p_final.txt", p)
    if verbose:
        print(f"[SIMPLE] Wrote: {case_dir}/U_final.txt, {case_dir}/p_final.txt")

    return U, p


if __name__ == "__main__":
    # Example run:
    # Make sure write_lid_driven_cavity_case.py has created the case directory first.
    simple_solve(
        case_dir="cavity_case",
        n_outer=100,
        alphaU=0.7,
        gs_sweeps=200,
        tol_div_inf=1e-8,
        tol_dU_inf=1e-10,
        ref_cell=0,
        verbose=True,
    )