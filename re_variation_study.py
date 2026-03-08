"""
re_variation_study.py
=====================
Step 5 (Report): Re number variation study
Re = 100, 200, 400, 1000

For each Re:
  - Run solver (or load if cavity_{NX}x{NY}_Re{Re}/U_final.txt exists)
  - Plot velocity magnitude
  - Plot centerline u(y) and v(x)
  - Record vortex core position, convergence iters

Produces:
  analysis_output/velocity_Re{Re}.png
  analysis_output/centerline_Re_comparison.png
  analysis_output/re_study_summary.txt

Usage:
  python re_variation_study.py

Note: Higher Re takes more iterations and may not fully converge.
      Re=1000 with 40x40 may show divergence — reduce alphaU if needed.
"""

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
)

# ── Ghia data (Re=100 only — for reference line on plots) ─────────────────────
GHIA_Y_U = np.array([1.,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,
                     0.5,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.])
GHIA_U   = np.array([1.,0.8412,0.7887,0.7372,0.6872,0.2315,0.00332,-0.1364,
                     -0.2058,-0.2109,-0.1566,-0.1015,-0.06434,-0.04775,-0.04192,-0.03717,0.])

# ── Re study config ────────────────────────────────────────────────────────────
RE_LIST  = [100, 200, 400, 1000]
NX = NY  = 40        # fixed grid for Re comparison (keep mesh constant)
N_OUTER  = 800       # more iterations for higher Re
OUT_DIR  = "analysis_output"

# Under-relaxation per Re (higher Re needs more conservative relaxation)
RELAX = {
    100:  (0.7, 0.3),
    200:  (0.6, 0.3),
    400:  (0.5, 0.2),
    1000: (0.4, 0.15),
}
COLORS = {100: "#1f77b4", 200: "#ff7f0e", 400: "#2ca02c", 1000: "#d62728"}
os.makedirs(OUT_DIR, exist_ok=True)


def normalize(cc):
    x, y = cc[:, 0], cc[:, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y


def extract_centerline(x, y, field, axis):
    if axis == "x":
        x_star = x[np.argmin(np.abs(x - 0.5))]
        mask = np.isclose(x, x_star)
        coord, vals = y[mask], field[mask]
    else:
        y_star = y[np.argmin(np.abs(y - 0.5))]
        mask = np.isclose(y, y_star)
        coord, vals = x[mask], field[mask]
    idx = np.argsort(coord)
    return coord[idx], vals[idx]


def find_vortex_core(x, y, Ux, Uy):
    """
    Approximate vortex core as location of minimum velocity magnitude,
    restricted to the interior region to avoid no-slip wall cells (|U|≈0).
    Interior region restricted to y ∈ (0.35, 0.98), x ∈ (0.05, 0.95),
    consistent with re_variation_analysis.py.
    """
    magU = np.sqrt(Ux**2 + Uy**2)
    mask = (y > 0.35) & (y < 0.98) & (x > 0.05) & (x < 0.95)
    idx  = np.where(mask)[0]
    local_min = idx[np.argmin(magU[idx])]
    return float(x[local_min]), float(y[local_min])


def solve_for_re(Re):
    aU, aP = RELAX[Re]
    case   = f"cavity_{NX}x{NY}_Re{Re}"
    u_path = os.path.join(case, "U_final.txt")
    p_path = os.path.join(case, "p_final.txt")
    h_path = os.path.join(case, "div_hist.txt")

    if os.path.exists(u_path) and os.path.exists(p_path):
        print(f"  [Load] Re={Re} from '{case}'")
        U  = np.loadtxt(u_path)
        p  = np.loadtxt(p_path)
        dh = np.loadtxt(h_path) if os.path.exists(h_path) else None
        n_conv = len(dh) if dh is not None else "?"
        mesh = Mesh.from_folder(os.path.join(case, "mesh"))
        return mesh, U, p, dh, n_conv

    print(f"  [Run] Re={Re}, alphaU={aU} (alphaP from RELAX unused) ...")
    if not os.path.exists(case):
        write_lid_driven_cavity_case(
            case_dir=case, nx=NX, ny=NY, d=0.1, U_lid=1.0, Re=Re)

    mesh = Mesh.from_folder(os.path.join(case, "mesh"))
    bc   = json.loads(open(f"{case}/bc.json").read())
    par  = json.loads(open(f"{case}/params.json").read())
    nu   = float(par["nu"])
    bcU  = bc["U"]
    bcp  = bc.get("p", None)
    if bcp:
        bcp = {k: v for k, v in bcp.items()
               if k in {pt.name for pt in mesh.patches}}

    U = np.loadtxt(f"{case}/0/U.txt")
    p = np.loadtxt(f"{case}/0/p.txt")

    div_hist, dU_hist = [], []
    n_conv = N_OUTER
    p_prime_init = np.zeros(len(mesh.cells))   # warm-start buffer

    for it in range(1, N_OUTER + 1):
        if it % 100 == 0:
            print(f"    iter {it}", flush=True)
        U_old = U.copy()
        Ustar, aP_x, aP_y = momentum_predictor(
            mesh, U, nu, bcU, alphaU=aU, gs_sweeps=200)
        Fpre    = compute_face_flux_linear(mesh, Ustar, bcU)
        p_prime = solve_pressure_equation(
            mesh, Fpre, aP_x, aP_y, alphaU=aU, bc_p=bcp,
            ref_cell=0, gs_sweeps=200,
            p_init=p_prime_init)            # warm-start
        p_prime_init = p_prime.copy()       # update for next iteration
        Fcorr = correct_face_flux(mesh, Fpre, p_prime, aP_x, aP_y)
        U     = correct_cell_velocity(mesh, Ustar, p_prime, aP_x, aP_y, bc_p=bcp)
        p    += p_prime          # full pressure update: p_new = p_old + p_prime

        div_inf = float(np.max(np.abs(divergence_of_face_flux(mesh, Fcorr))))
        dU_inf  = float(np.max(np.abs(U - U_old)))
        div_hist.append(div_inf)
        dU_hist.append(dU_inf)

        if div_inf < 1e-7 and dU_inf < 1e-9:
            n_conv = it
            print(f"    Converged at iter {it}")
            break

    np.savetxt(f"{case}/U_final.txt", U)
    np.savetxt(f"{case}/p_final.txt", p)
    dh = np.array(div_hist)
    np.savetxt(os.path.join(case, "div_hist.txt"), dh)
    np.savetxt(os.path.join(case, "dU_hist.txt"),  np.array(dU_hist))
    return mesh, U, p, dh, n_conv


def main():
    all_results = {}
    summary_lines = ["Re Study Summary", "=" * 60]

    for Re in RE_LIST:
        print(f"\n{'='*50}\nRe = {Re}")
        mesh, U, p, div_hist, n_conv = solve_for_re(Re)
        cc = np.array(mesh.cell_centers)
        x, y = normalize(cc)
        Ux, Uy = U[:, 0], U[:, 1]
        magU = np.sqrt(Ux**2 + Uy**2)

        vx, vy = find_vortex_core(x, y, Ux, Uy)
        y_line, u_line = extract_centerline(x, y, Ux, "x")
        x_line, v_line = extract_centerline(x, y, Uy, "y")

        all_results[Re] = {
            "mesh": mesh, "x": x, "y": y,
            "Ux": Ux, "Uy": Uy, "magU": magU, "p": p,
            "y_line": y_line, "u_line": u_line,
            "x_line": x_line, "v_line": v_line,
            "vortex": (vx, vy), "n_conv": n_conv,
            "div_hist": div_hist,
        }

        summary_lines.append(
            f"Re={Re:>5d}: vortex core≈({vx:.3f},{vy:.3f}), "
            f"converged@iter={n_conv}"
        )

        # Velocity magnitude plot per Re
        tri = mtri.Triangulation(x, y)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        cf = ax.tricontourf(tri, magU, levels=30, cmap="viridis")
        plt.colorbar(cf, ax=ax, label="|U|")
        ax.plot(vx, vy, "r*", ms=12, label=f"Core≈({vx:.2f},{vy:.2f})")
        ax.set_aspect("equal")
        ax.set_title(f"Velocity magnitude — Re={Re}, {NX}×{NY}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"velocity_Re{Re}.png")
        plt.savefig(fname, dpi=200); plt.close()
        print(f"  [Saved] {fname}")

    # ── Combined centerline comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for Re, d in all_results.items():
        axes[0].plot(d["u_line"], d["y_line"], color=COLORS[Re],
                     lw=1.8, label=f"Re={Re}")
        axes[1].plot(d["x_line"], d["v_line"], color=COLORS[Re],
                     lw=1.8, label=f"Re={Re}")

    axes[0].plot(GHIA_U, GHIA_Y_U, "ko", ms=4, label="Ghia Re=100")
    axes[0].set_title("u(y) at x = 0.5 — Re comparison")
    axes[0].set_xlabel("u"); axes[0].set_ylabel("y")
    axes[0].legend(); axes[0].grid(True)

    axes[1].set_title("v(x) at y = 0.5 — Re comparison")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("v")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "centerline_Re_comparison.png")
    plt.savefig(fname, dpi=200); plt.close()
    print(f"[Saved] {fname}")

    # ── Convergence comparison across Re ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for Re, d in all_results.items():
        dh = d["div_hist"]
        if dh is not None and len(dh) > 0:
            ax.semilogy(np.arange(1, len(dh)+1), dh,
                        color=COLORS[Re], lw=1.8, label=f"Re={Re}")
    ax.set_xlabel("Outer iteration"); ax.set_ylabel(r"$\|\nabla\cdot F\|_\infty$")
    ax.set_title(f"Convergence — Re comparison ({NX}×{NY} grid)")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "convergence_Re_comparison.png")
    plt.savefig(fname, dpi=200); plt.close()
    print(f"[Saved] {fname}")

    # ── Vortex core trajectory plot ──────────────────────────────────────────
    re_vals = list(all_results.keys())
    vx_vals = [all_results[r]["vortex"][0] for r in re_vals]
    vy_vals = [all_results[r]["vortex"][1] for r in re_vals]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(vx_vals, vy_vals, c=re_vals, cmap="plasma",
                    s=120, zorder=5, edgecolors="k")
    for re, vx, vy in zip(re_vals, vx_vals, vy_vals):
        ax.annotate(f"Re={re}", (vx, vy),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.colorbar(sc, ax=ax, label="Re")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Primary vortex core position vs Re ({NX}×{NY} grid)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "vortex_core_vs_Re.png")
    plt.savefig(fname, dpi=200); plt.close()
    print(f"[Saved] {fname}")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary_lines += [
        "",
        "Expected physics with increasing Re:",
        "  Re=100 : single primary vortex, centre ~(0.62, 0.74)",
        "  Re=200 : primary vortex shifts right/down, secondary vortex larger",
        "  Re=400 : secondary vortices more prominent, slower convergence",
        "  Re=1000: tertiary vortex appears, upwind diffusion may smear details",
        "",
        "Convergence trend: higher Re → more outer iterations needed",
        "(stronger convection → weaker diagonal dominance → slower GS)",
    ]
    tbl_path = os.path.join(OUT_DIR, "re_study_summary.txt")
    with open(tbl_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"[Saved] {tbl_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()