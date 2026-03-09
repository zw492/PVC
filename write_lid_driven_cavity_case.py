# Task 3
# write_lid_driven_cavity_case.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from face_addressed_mesh_2d import generate_cartesian_mesh_files, Mesh

def write_lid_driven_cavity_case(
    case_dir: str,
    nx: int = 40,
    ny: int = 40,
    d: float = 0.1,
    U_lid: float = 1.0,
    Re: float = 100.0,
) -> None:
    """
    Create a complete 2D lid-driven cavity 'case' folder:
      case_dir/
        mesh/points faces cells boundary
        0/U.txt 0/p.txt
        params.json
        bc.json
    Notes:
        Pressure is kinematic pressure p = P/rho.
        Pressure BC is zeroGradient on all walls; a reference cell is fixed to remove nullspace.
    """
    case = Path(case_dir)
    mesh_dir = case / "mesh"
    t0_dir = case / "0"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    t0_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mesh files (face-addressed) with physical size d x d
    patch_names = ("left", "right", "bottom", "top")
    generate_cartesian_mesh_files(mesh_dir, nx=nx, ny=ny, Lx=d, Ly=d, patch_names=patch_names)

    # 2) Read mesh (to know nCells and patches)
    m = Mesh.from_folder(mesh_dir)
    nC = len(m.cells)

    # 3) Physical parameters
    if Re <= 0:
        raise ValueError("Re must be > 0")
    nu = U_lid * d / Re  # m^2/s

    params = {
        "d": float(d),
        "Re": float(Re),
        "U_lid": float(U_lid),
        "nu": float(nu),
        "rho": 1.0,  # p is kinematic pressure
    }
    (case / "params.json").write_text(json.dumps(params, indent=2))

    # 4) Initial fields (cell-centre)
    U = np.zeros((nC, 2), dtype=float)  # [Ux, Uy] per cell
    p = np.zeros((nC,), dtype=float)

    np.savetxt(t0_dir / "U.txt", U, fmt="%.8e")
    np.savetxt(t0_dir / "p.txt", p, fmt="%.8e")

    # 5) Boundary conditions
    bc = {
        "U": {
            "left":   {"type": "fixedValue", "value": [0.0, 0.0]},
            "right":  {"type": "fixedValue", "value": [0.0, 0.0]},
            "bottom": {"type": "fixedValue", "value": [0.0, 0.0]},
            "top":    {"type": "fixedValue", "value": [float(U_lid), 0.0]},
        },
        "p": {
            "left":   {"type": "zeroGradient"},
            "right":  {"type": "zeroGradient"},
            "bottom": {"type": "zeroGradient"},
            "top":    {"type": "zeroGradient"},
            "reference": {"type": "fixedCell", "cell": 0, "value": 0.0},
        },
    }
    (case / "bc.json").write_text(json.dumps(bc, indent=2))

    # Confirmation
    print(f"   Case written to: {case.resolve()}")
    print(f"   Mesh: {nx}x{ny}, d={d}, Re={Re}, nu={nu}")
    print(f"   Cells: {nC}, Patches: {[p.name for p in m.patches]}")
    print(f"   Mesh files: {(mesh_dir / 'points').name}, {(mesh_dir / 'faces').name}, {(mesh_dir / 'cells').name}, {(mesh_dir / 'boundary').name}")


if __name__ == "__main__":
    # Default: create the 40x40 cavity case used by Master_Analysis.py (CASE_DIR[40])
    write_lid_driven_cavity_case("cavity_case", nx=40, ny=40, d=0.1, U_lid=1.0, Re=100.0)

    # Optional: also create a 20x20 case for C2 refinement check
    write_lid_driven_cavity_case("cavity_case_20", nx=20, ny=20, d=0.1, U_lid=1.0, Re=100.0)

