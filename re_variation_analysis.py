"""
re_variation_analysis.py
Post-processing for Re variation study (Re=100/200/400/1000), all on 40×40 grid.

Generates in re_analysis_output/:
  centerlines_Re{N}.png     u(y) and v(x) vs Ghia (where available)
  convergence_Re{N}.png     div_hist and dU_hist per Re
  convergence_all_Re.png    div_hist all Re on one plot
  vortex_centers.txt        primary vortex centre x,y per Re
Requires: cavity_40x40_Re100/, cavity_40x40_Re200/,
          cavity_40x40_Re400/, cavity_40x40_Re1000/
          each containing U_final.txt, div_hist.txt, dU_hist.txt
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from face_addressed_mesh_2d import Mesh

# Config
RE_LIST   = [100, 200, 400, 1000]
CASE_DIR  = {re: f"cavity_40x40_Re{re}" for re in RE_LIST}
OUT_DIR   = "re_analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {100: "#1f77b4", 200: "#ff7f0e", 400: "#2ca02c", 1000: "#d62728"}

# Ghia (1982) reference data
# Re=100
G100_Y = np.array([1.0,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,
                   0.5,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G100_U = np.array([1.0,0.8412,0.7887,0.7372,0.6872,0.2315,0.00332,-0.1364,
                   -0.2058,-0.2109,-0.1566,-0.1015,-0.06434,-0.04775,-0.04192,-0.03717,0.0])
G100_X = np.array([1.0,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5,
                   0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G100_V = np.array([0.0,-0.05906,-0.07391,-0.08864,-0.2453,-0.2245,-0.1691,
                   0.05454,0.1751,0.1753,0.1608,0.1232,0.1089,0.1009,0.09233,0.0])

# Re=400
G400_Y = np.array([1.0,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,
                   0.5,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G400_U = np.array([1.0,0.7573,0.6859,0.6227,0.5662,0.2877,0.1183,-0.1263,
                   -0.2264,-0.2340,-0.1871,-0.1216,-0.0761,-0.0556,-0.0478,-0.0419,0.0])
G400_X = np.array([1.0,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5,
                   0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G400_V = np.array([0.0,-0.12146,-0.15663,-0.19254,-0.3869,-0.3838,-0.2973,
                   0.09060,0.3306,0.3273,0.2803,0.1973,0.1702,0.1579,0.1438,0.0])

# Re=1000
G1000_Y = np.array([1.0,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,
                    0.5,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G1000_U = np.array([1.0,0.6590,0.5767,0.5102,0.4572,0.3333,0.3780,0.2795,
                    0.1402,0.0272,-0.3272,-0.2293,-0.1175,-0.0807,-0.0694,-0.0608,0.0])
G1000_X = np.array([1.0,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5,
                    0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0])
G1000_V = np.array([0.0,-0.2174,-0.2920,-0.3769,-0.5155,-0.4253,-0.2973,
                    0.1794,0.4034,0.4307,0.3507,0.2286,0.1925,0.1788,0.1620,0.0])

GHIA = {
    100:  (G100_Y,  G100_U,  G100_X,  G100_V),
    400:  (G400_Y,  G400_U,  G400_X,  G400_V),
    1000: (G1000_Y, G1000_U, G1000_X, G1000_V),
}

# Utilities
def normalize(cc):
    x, y = cc[:, 0], cc[:, 1]
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return x, y

def centerline(x, y, field, axis):
    if axis == "x":
        xs = x[np.argmin(np.abs(x - 0.5))]
        mask = np.isclose(x, xs)
        coord, vals = y[mask], field[mask]
    else:
        ys = y[np.argmin(np.abs(y - 0.5))]
        mask = np.isclose(y, ys)
        coord, vals = x[mask], field[mask]
    idx = np.argsort(coord)
    return coord[idx], vals[idx]

def find_vortex_center(x, y, U):
    """
    Estimate primary vortex centre as the cell with minimum velocity magnitude,
    restricted to the interior region (away from all four walls) where the
    primary vortex resides for all Re studied.
    Interior region restricted to y ∈ (0.35, 0.98), x ∈ (0.05, 0.95) to avoid
    no-slip wall cells (|U|≈0) which would otherwise dominate the minimum search.
    """
    magU = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    mask = (y > 0.35) & (y < 0.98) & (x > 0.05) & (x < 0.95)
    idx  = np.where(mask)[0]
    local_min = idx[np.argmin(magU[idx])]
    return float(x[local_min]), float(y[local_min])

def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")

# Load all data
data = {}
for re in RE_LIST:
    cdir = CASE_DIR[re]
    print(f"Loading Re={re} from '{cdir}' ...")
    U        = np.loadtxt(f"{cdir}/U_final.txt")
    div_hist = np.loadtxt(f"{cdir}/div_hist.txt")
    dU_hist  = np.loadtxt(f"{cdir}/dU_hist.txt")
    mesh     = Mesh.from_folder(f"{cdir}/mesh")
    cc       = np.array(mesh.cell_centers)
    x, y     = normalize(cc)
    data[re] = dict(U=U, div=div_hist, dU=dU_hist, x=x, y=y)

# 1. Centerline plots per Re
print("\nPlotting centerlines ...")
for re in RE_LIST:
    d   = data[re]
    y_l, u_l = centerline(d["x"], d["y"], d["U"][:, 0], "x")
    x_l, v_l = centerline(d["x"], d["y"], d["U"][:, 1], "y")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Centreline profiles — Re={re}, 40×40", fontsize=12)

    axes[0].plot(u_l, y_l, color=COLORS[re], lw=2, label=f"Re={re}")
    axes[1].plot(x_l, v_l, color=COLORS[re], lw=2, label=f"Re={re}")

    if re in GHIA:
        gy_u, gu, gx_v, gv = GHIA[re]
        axes[0].plot(gu, gy_u, "ko", ms=4, label="Ghia 1982")
        axes[1].plot(gx_v, gv, "ko", ms=4, label="Ghia 1982")

    for ax, xl, yl, tl in zip(axes,
                               ["u", "x"], ["y", "v"],
                               ["u(y) at x=0.5", "v(x) at y=0.5"]):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(tl)
        ax.legend(); ax.grid(True, ls="--", alpha=0.5)

    _save(fig, os.path.join(OUT_DIR, f"centerlines_Re{re}.png"))

# 2a. Convergence per Re
print("Plotting convergence per Re ...")
for re in RE_LIST:
    d     = data[re]
    iters = np.arange(1, len(d["div"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f"SIMPLE outer-loop convergence — Re={re}, 40×40", fontsize=12)

    ax1.semilogy(iters, d["div"], color=COLORS[re], lw=1.5)
    ax1.set_ylabel(r"$\|\nabla\cdot F_{corr}\|_\infty$", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    ax2.semilogy(iters, np.clip(d["dU"], 1e-20, None), color=COLORS[re], lw=1.5)
    ax2.set_ylabel(r"$\|\Delta U\|_\infty$", fontsize=11)
    ax2.set_xlabel("Outer iteration", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    _save(fig, os.path.join(OUT_DIR, f"convergence_Re{re}.png"))

# 2b. All Re div_hist on one plot
print("Plotting all-Re convergence comparison ...")
fig, ax = plt.subplots(figsize=(9, 5))
for re in RE_LIST:
    d = data[re]
    ax.semilogy(np.arange(1, len(d["div"]) + 1), d["div"],
                color=COLORS[re], lw=1.8, label=f"Re={re}")
ax.set_xlabel("Outer iteration", fontsize=11)
ax.set_ylabel(r"$\|\nabla\cdot F_{corr}\|_\infty$", fontsize=11)
ax.set_title("Convergence comparison — all Re, 40×40")
ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.4)
_save(fig, os.path.join(OUT_DIR, "convergence_all_Re.png"))

# 3. Vortex centre table
print("Computing vortex centres ...")
lines = [
    "=" * 52,
    "  Primary Vortex Centre — 40×40 grid",
    "=" * 52,
    f"{'Re':>6}  {'x_centre':>10}  {'y_centre':>10}  {'Ghia x':>8}  {'Ghia y':>8}",
    "-" * 52,
]

# Ghia vortex centre reference values (from Ghia 1982 Table 1, Re=100/400/1000)
GHIA_VORTEX = {100: (0.6172, 0.7344), 400: (0.5547, 0.6055), 1000: (0.5313, 0.5625)}

for re in RE_LIST:
    d      = data[re]
    xc, yc = find_vortex_center(d["x"], d["y"], d["U"])
    if re in GHIA_VORTEX:
        gx, gy = GHIA_VORTEX[re]
        lines.append(f"{re:>6}  {xc:>10.4f}  {yc:>10.4f}  {gx:>8.4f}  {gy:>8.4f}")
    else:
        lines.append(f"{re:>6}  {xc:>10.4f}  {yc:>10.4f}  {'N/A':>8}  {'N/A':>8}")

lines += ["=" * 52,
          "Note: vortex centre = cell of minimum |U| with y>0.4"]

path = os.path.join(OUT_DIR, "vortex_centers.txt")
with open(path, "w") as f:
    f.write("\n".join(lines))
print(f"  [Saved] {path}")
print("\n".join(lines))


print(f"\nDone. All outputs in '{OUT_DIR}/'")
