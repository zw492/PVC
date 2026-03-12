"""
test20x20.py
Temporary single-grid sanity check: runs the corrected SIMPLE solver on the
20×20 cavity (Re=100, n_outer=400) and produces the same diagnostic outputs
as Master_Analysis.py would for that grid, such that I can verify correctness
before committing the full cluster run.

Produces in test_20x20_output/:
  convergence_outer_20x20.png
  convergence_inner_GS.png   (schematic, same style as Master_Analysis)
  error_table_20x20.txt
  iterations_table_20x20.txt
  matrix_coefficients_20x20.txt
  pressure_field_20x20.png
  velocity_field_20x20.png
  velocity_magnitude_20x20_Re100.png
"""

from __future__ import annotations
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from face_addressed_mesh_2d import Mesh
from write_lid_driven_cavity_case import write_lid_driven_cavity_case
from momentum_predictor import momentum_predictor
from pressure_correction import (
    compute_face_flux_linear, solve_pressure_equation,
    correct_face_flux, correct_cell_velocity,
    divergence_of_face_flux,
    assemble_laplace_variable_gamma, pin_pressure_reference,
)

# Config
N        = 20
RE       = 100
ALPHA_U  = 0.7
GS_SWEEPS = 200
N_OUTER  = 400
TOL_DIV  = 1e-5
TOL_DU   = 1e-6
CASE_DIR = "cavity_20x20_Re100"
OUT_DIR  = "test_20x20_output"

# Ghia (1982) Re=100 reference data
GHIA_Y_U = np.array([1.0000,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,
                     0.5000,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0000])
GHIA_U   = np.array([1.0000,0.8412,0.7887,0.7372,0.6872,0.2315,0.00332,-0.1364,
                     -0.2058,-0.2109,-0.1566,-0.1015,-0.06434,-0.04775,-0.04192,-0.03717,0.0000])
GHIA_X_V = np.array([1.0000,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5000,
                     0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0000])
GHIA_V   = np.array([0.0000,-0.05906,-0.07391,-0.08864,-0.2453,-0.2245,-0.1691,
                     0.05454,0.1751,0.1753,0.1608,0.1232,0.1089,0.1009,0.09233,0.0000])

os.makedirs(OUT_DIR, exist_ok=True)


# Utilities

def normalize(cc):
    x, y = cc[:, 0], cc[:, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def extract_centerline(x, y, field, axis):
    if axis == "x":
        xs = x[np.argmin(np.abs(x - 0.5))]
        mask = np.isclose(x, xs); coord, vals = y[mask], field[mask]
    else:
        ys = y[np.argmin(np.abs(y - 0.5))]
        mask = np.isclose(y, ys); coord, vals = x[mask], field[mask]
    idx = np.argsort(coord)
    return coord[idx], vals[idx]


def _save(fig, fname):
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {fname}")


# Solver

def run_solver():
    if not os.path.exists(CASE_DIR):
        print(f"Writing case files for {N}×{N} …")
        write_lid_driven_cavity_case(
            case_dir=CASE_DIR, nx=N, ny=N, d=0.1, U_lid=1.0, Re=RE)

    mesh   = Mesh.from_folder(f"{CASE_DIR}/mesh")
    bc     = json.loads(open(f"{CASE_DIR}/bc.json").read())
    params = json.loads(open(f"{CASE_DIR}/params.json").read())
    nu     = float(params["nu"])
    bcU    = bc["U"]
    bcp    = bc.get("p", None)
    if bcp:
        bcp = {k: v for k, v in bcp.items()
               if k in {pt.name for pt in mesh.patches}}

    U = np.loadtxt(f"{CASE_DIR}/0/U.txt")
    p = np.loadtxt(f"{CASE_DIR}/0/p.txt")

    div_hist, dU_hist = [], []
    n_conv = N_OUTER
    nC = len(mesh.cells)
    p_prime_init = np.zeros(nC)   # warm-start buffer for pressure equation

    print(f"Running SIMPLE: {N}×{N}, Re={RE}, alphaU={ALPHA_U}, n_outer={N_OUTER}")
    for it in range(1, N_OUTER + 1):
        if it % 50 == 0:
            print(f"  iter {it}/{N_OUTER}", flush=True)

        U_old = U.copy()

        # Step 5.1 Momentum predictor
        Ustar, aP_x, aP_y = momentum_predictor(
            mesh, U, nu, bcU, alphaU=ALPHA_U, gs_sweeps=GS_SWEEPS)

        # Step 5.2 Precursor face flux
        Fpre = compute_face_flux_linear(mesh, Ustar, bcU)

        # Step 5.3 Pressure correction (gamma = 1/aP)
        p_prime = solve_pressure_equation(
            mesh, Fpre, aP_x, aP_y, bc_p=bcp,
            ref_cell=0, gs_sweeps=GS_SWEEPS,
            p_init=p_prime_init)          # warm-start: reuse previous p_prime
        p_prime_init = p_prime.copy()     # update for next iteration

        # Step 5.4 Flux and velocity correction
        Fcorr = correct_face_flux(mesh, Fpre, p_prime, aP_x, aP_y)
        U     = correct_cell_velocity(mesh, Ustar, p_prime, aP_x, aP_y, bc_p=bcp)

        # Pressure accumulation: full update
        p += p_prime

        div_inf = float(np.max(np.abs(divergence_of_face_flux(mesh, Fcorr))))
        dU_inf  = float(np.max(np.abs(U - U_old)))
        div_hist.append(div_inf)
        dU_hist.append(dU_inf)

        if div_inf < TOL_DIV and dU_inf < TOL_DU:
            n_conv = it
            print(f"  ✓ Converged at iter {it}")
            break

    np.savetxt(f"{CASE_DIR}/U_final.txt", U)
    np.savetxt(f"{CASE_DIR}/p_final.txt", p)
    np.savetxt(f"{CASE_DIR}/div_hist.txt", div_hist)
    np.savetxt(f"{CASE_DIR}/dU_hist.txt",  dU_hist)

    print(f"\nPressure field sanity check (Jasak expects ~ -4 to +4 for 20×20):")
    print(f"  p_min = {p.min():.4f}   p_max = {p.max():.4f}   range = {p.max()-p.min():.4f}")

    return mesh, U, p, np.array(div_hist), np.array(dU_hist), n_conv


# Plots

def plot_outer_convergence(div_hist, dU_hist):
    iters = np.arange(1, len(div_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.semilogy(iters, div_hist, color="#1f77b4", lw=1.5)
    ax1.axhline(1e-5, color="red", ls=":", lw=1, label="tol")
    ax1.set_ylabel(r"$\|\nabla \cdot \mathbf{F}_{corr}\|_\infty$", fontsize=12)
    ax1.set_title(f"SIMPLE outer loop convergence — {N}×{N}, Re={RE}")
    ax1.legend(fontsize=9); ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax2.semilogy(iters, np.clip(dU_hist, 1e-20, None), color="#ff7f0e", lw=1.5)
    ax2.set_ylabel(r"$\|\Delta \mathbf{U}\|_\infty$", fontsize=12)
    ax2.set_xlabel("Outer iteration", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    _save(fig, os.path.join(OUT_DIR, f"convergence_outer_{N}x{N}.png"))


def plot_inner_gs_schematic():
    sweeps = np.arange(0, 201)
    def decay(r0, rf, n=200):
        if r0 == 0 or rf == 0: return np.zeros_like(sweeps, dtype=float)
        lam = -np.log(rf / r0) / n
        return r0 * np.exp(-lam * sweeps)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(sweeps, decay(0.01265, 1.6e-18), color="#1f77b4", lw=2,
                label="Ux — GS converges (iter 1)")
    ax.semilogy(sweeps, decay(0.00632, 1.6e-18), color="#1f77b4", lw=1.5, ls="--",
                label="Ux — GS converges (iter 2)")
    ax.axhline(1e-16, color="#ff7f0e", lw=2, ls="-",
               label="Uy — residual ≈ 0 (no driving term without ∇p)")
    ax.axvline(200, color="red", ls=":", lw=1.5, label="200 sweeps limit")
    ax.set_xlabel("GS sweep within one outer iteration", fontsize=11)
    ax.set_ylabel("GS residual (unnormalised)", fontsize=11)
    ax.set_title("Inner GS convergence — momentum predictor\n"
                 "Uy residual ≈ 0: no pressure gradient → no driving term (Task 10)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, which="both", ls="--", alpha=0.4)
    _save(fig, os.path.join(OUT_DIR, "convergence_inner_GS.png"))


def plot_pressure(mesh, p):
    cc = np.array(mesh.cell_centers)
    x, y = normalize(cc)
    tri  = mtri.Triangulation(x, y)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cf = ax.tricontourf(tri, p, levels=30, cmap="RdBu_r")
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label("p [m²/s²]")
    cs   = ax.tricontour(tri, p, levels=10, colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.3f")
    ax.set_aspect("equal")
    ax.set_title(f"Pressure field (Re={RE}, {N}×{N})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    _save(fig, os.path.join(OUT_DIR, f"pressure_field_{N}x{N}.png"))


def plot_velocity(mesh, U):
    cc = np.array(mesh.cell_centers)
    x, y   = normalize(cc)
    Ux, Uy = U[:, 0], U[:, 1]
    magU   = np.sqrt(Ux**2 + Uy**2)
    tri    = mtri.Triangulation(x, y)
    step   = max(1, N // 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cf = axes[0].tricontourf(tri, magU, levels=30, cmap="viridis")
    plt.colorbar(cf, ax=axes[0], label="|U| [m/s]")
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Velocity magnitude (Re={RE}, {N}×{N})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].tricontourf(tri, magU, levels=30, cmap="viridis", alpha=0.6)
    axes[1].quiver(x[::step], y[::step], Ux[::step], Uy[::step],
                   scale=15, width=0.003, color="white", alpha=0.85)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Velocity vectors (Re={RE}, {N}×{N})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    _save(fig, os.path.join(OUT_DIR, f"velocity_field_{N}x{N}.png"))

    # Separate velocity magnitude plot (matches Master_Analysis naming)
    fig2, ax2 = plt.subplots(figsize=(5.5, 5))
    cf2 = ax2.tricontourf(tri, magU, levels=30, cmap="viridis")
    plt.colorbar(cf2, ax=ax2, label="|U| [m/s]")
    ax2.set_aspect("equal")
    ax2.set_title(f"Velocity magnitude (Re={RE}, {N}×{N})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    _save(fig2, os.path.join(OUT_DIR, f"velocity_magnitude_{N}x{N}_Re{RE}.png"))


def plot_centerlines(x, y, Ux, Uy):
    y_line, u_line = extract_centerline(x, y, Ux, "x")
    x_line, v_line = extract_centerline(x, y, Uy, "y")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(u_line, y_line, color="#1f77b4", lw=2, label=f"{N}×{N}")
    axes[0].plot(GHIA_U, GHIA_Y_U, "ko", ms=4, label="Ghia 1982")
    axes[0].set_title("u(y) at x = 0.5"); axes[0].set_xlabel("u"); axes[0].set_ylabel("y")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(x_line, v_line, color="#1f77b4", lw=2, label=f"{N}×{N}")
    axes[1].plot(GHIA_X_V, GHIA_V, "ko", ms=4, label="Ghia 1982")
    axes[1].set_title("v(x) at y = 0.5"); axes[1].set_xlabel("x"); axes[1].set_ylabel("v")
    axes[1].legend(); axes[1].grid(True)
    _save(fig, os.path.join(OUT_DIR, f"centerline_{N}x{N}_Re{RE}.png"))

    return y_line, u_line, x_line, v_line


def write_error_table(y_line, u_line, x_line, v_line):
    u_num = np.interp(GHIA_Y_U[::-1], y_line, u_line)[::-1]
    v_num = np.interp(GHIA_X_V[::-1], x_line, v_line)[::-1]
    eu, ev = u_num - GHIA_U, v_num - GHIA_V
    lines = [
        "=" * 60,
        f"  Error vs Ghia (1982) — Re={RE}, {N}×{N}",
        "=" * 60,
        f"  u_L2   = {np.sqrt(np.mean(eu**2)):.5f}",
        f"  u_Linf = {np.max(np.abs(eu)):.5f}",
        f"  v_L2   = {np.sqrt(np.mean(ev**2)):.5f}",
        f"  v_Linf = {np.max(np.abs(ev)):.5f}",
        "",
        f"  u_max (num) = {u_num.max():.5f}   u_max (Ghia) = {GHIA_U.max():.5f}",
        f"  u_min (num) = {u_num.min():.5f}   u_min (Ghia) = {GHIA_U.min():.5f}",
        f"  v_max (num) = {v_num.max():.5f}   v_max (Ghia) = {GHIA_V.max():.5f}",
        f"  v_min (num) = {v_num.min():.5f}   v_min (Ghia) = {GHIA_V.min():.5f}",
        "=" * 60,
    ]
    path = os.path.join(OUT_DIR, "error_table_20x20.txt")
    with open(path, "w") as f: f.write("\n".join(lines))
    print(f"  [Saved] {path}")
    print("\n".join(lines))


def write_iterations_table(div_hist, n_conv):
    lines = [
        "Grid Convergence Summary (Re=100, 20×20)",
        "=" * 45,
        f"  Outer iterations to tol: {n_conv}",
        f"  Final ||div(Fcorr)||_inf : {div_hist[-1]:.3e}",
        "=" * 45,
    ]
    path = os.path.join(OUT_DIR, "iterations_table_20x20.txt")
    with open(path, "w") as f: f.write("\n".join(lines))
    print(f"  [Saved] {path}")
    print("\n".join(lines))


def report_matrix_coefficients(mesh, U):
    bc     = json.loads(open(f"{CASE_DIR}/bc.json").read())
    params = json.loads(open(f"{CASE_DIR}/params.json").read())
    nu     = float(params["nu"])
    bcU    = bc["U"]
    bcp    = bc.get("p", None)
    if bcp:
        bcp = {k: v for k, v in bcp.items()
               if k in {pt.name for pt in mesh.patches}}

    nC = len(mesh.cells)
    cc = np.array(mesh.cell_centers)

    cx = (cc[:, 0].min() + cc[:, 0].max()) / 2
    cy = (cc[:, 1].min() + cc[:, 1].max()) / 2
    dist  = np.sqrt((cc[:, 0] - cx)**2 + (cc[:, 1] - cy)**2)
    c_int = int(np.argmin(dist))
    c_lid = None
    for patch in mesh.patches:
        if patch.name.lower() in ("top", "lid", "top_wall"):
            mid   = len(patch.face_ids) // 2
            c_lid = mesh.face_owner[patch.face_ids[mid]]
            break
    if c_lid is None:
        c_lid = int(np.argmax(cc[:, 1]))

    Ustar, aP_x, aP_y = momentum_predictor(
        mesh, U, nu, bcU, alphaU=ALPHA_U, gs_sweeps=GS_SWEEPS)

    aP_        = 0.5 * (aP_x + aP_y)
    gamma_cell = 1.0 / aP_
    Fpre       = compute_face_flux_linear(mesh, Ustar, bcU)
    bcp_eff    = bcp if bcp else {pt.name: {"type": "zeroGradient"} for pt in mesh.patches}
    trip, b_l  = assemble_laplace_variable_gamma(mesh, gamma_cell, bcp_eff)
    b          = b_l + (-divergence_of_face_flux(mesh, Fpre))
    trip, b    = pin_pressure_reference(nC, trip, b, ref_cell=0)

    def pres_desc(cid, label):
        rc = {}
        for i, j, v in trip:
            if i == cid: rc[j] = rc.get(j, 0.0) + v
        diag = rc.get(cid, 0.0)
        od   = {j: v for j, v in rc.items() if j != cid}
        lines = [f"\n[{label}] cell={cid}, centre=({cc[cid,0]:.4f},{cc[cid,1]:.4f})",
                 f"  Diagonal a_P      = {diag:.6e}",
                 f"  RHS b             = {b[cid]:.6e}",
                 f"  Off-diagonal ({len(od)}):"]
        for j, v in sorted(od.items()):
            lines.append(f"    cell {j:>4d}: a_N = {v:.6e}  ({cc[j,0]:.4f},{cc[j,1]:.4f})")
        s = sum(od.values())
        lines += [f"  Σ off-diag        = {s:.6e}",
                  f"  a_P + Σ a_N       = {diag+s:.6e}  (≈0 for Neumann interior)"]
        return "\n".join(lines)

    out = [
        "=" * 70, "MATRIX COEFFICIENT REPORT — 20×20 TEST",
        f"Case: {CASE_DIR}  |  nu={nu}  |  alphaU={ALPHA_U}", "=" * 70,
        "\nTASK 8 — Momentum matrix", "-" * 70,
        f"  c_int = {c_int}  aP_x={aP_x[c_int]:.6e}  aP_y={aP_y[c_int]:.6e}",
        f"  c_lid = {c_lid}  aP_x={aP_x[c_lid]:.6e}  aP_y={aP_y[c_lid]:.6e}",
        "\nTASK 9 — Pressure matrix", "-" * 70,
        pres_desc(c_int, "INTERNAL CELL"),
        pres_desc(c_lid, "LID-ADJACENT (top wall)"),
        "\nNotes:",
        "  Momentum aP is the PRE-relaxation diagonal",
        "  Pressure gamma = 1/aP",
        "  Pressure off-diag = -gamma_f*|Sf|*delta_f  (always negative)",
        "  For all-Neumann BCs: sum(row) = 0 for interior cells",
    ]
    path = os.path.join(OUT_DIR, "matrix_coefficients_20x20.txt")
    with open(path, "w") as f: f.write("\n".join(out))
    print(f"  [Saved] {path}")


# Main

def main():
    print(f"\n{'='*60}")
    print(f"  test_20x20.py  —  {N}×{N}, Re={RE}, n_outer={N_OUTER}")
    print(f"{'='*60}\n")

    mesh, U, p, div_hist, dU_hist, n_conv = run_solver()
    cc = np.array(mesh.cell_centers)
    x, y   = normalize(cc)
    Ux, Uy = U[:, 0], U[:, 1]

    print("\n── Plotting fields ──")
    plot_pressure(mesh, p)
    plot_velocity(mesh, U)

    print("\n── Plotting convergence ──")
    plot_outer_convergence(div_hist, dU_hist)
    plot_inner_gs_schematic()

    print("\n── Centerlines + error table ──")
    y_line, u_line, x_line, v_line = plot_centerlines(x, y, Ux, Uy)
    write_error_table(y_line, u_line, x_line, v_line)

    print("\n── Iterations summary ──")
    write_iterations_table(div_hist, n_conv)

    print("\n── Matrix coefficients (Tasks 8 & 9) ──")
    report_matrix_coefficients(mesh, U)

    print(f"\n{'='*60}")
    print(f"  Done. Outputs in '{OUT_DIR}/'")
    print(f"{'='*60}")


if __name__ == "__main__":

    main()
