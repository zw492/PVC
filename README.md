# PVC: SIMPLE-Based Steady-State Incompressible Flow Solver

Pressure-Velocity Coupling: SIMPLE-Based Steady-State Solver. Adviser: Professor Hrvoje Jasak

A two-dimensional finite-volume solver for steady incompressible flow using the SIMPLE pressure–velocity coupling algorithm, implemented in pure Python/NumPy. Applied to the lid-driven square cavity benchmark and validated against Ghia et al. (1982).

---

## Repository Structure

```
.
├── face_addressed_mesh_2d.py       # Mesh class and Cartesian mesh generator
├── write_lid_driven_cavity_case.py # Case setup: mesh files, BCs, initial fields
├── sparse_cr.py                    # Compressed-Row sparse matrix and Gauss–Seidel solver
├── fv_scalar.py                    # FV operators: gradient, diffusion, upwind convection
├── momentum_predictor.py           # Momentum predictor with implicit under-relaxation
├── pressure_correction.py          # Pressure equation, flux correction, velocity correction
├── simple_solver.py                # Standalone SIMPLE outer loop (diagnostic)
├── test20x20.py                    # Single-grid sanity check (20×20, Re=100)
├── Master_Analysis.py              # Main driver: 20, 40, 80 grids, Re = 100
├── re_variation_study.py           # Re parametric driver: Re=100/200/400/1000
├── task6_7_11_12_driver.py         # Results driver for tasks 6, 7, 12
└── re_variation_analysis.py        # Post-processing for Re variation results
```

---

## Module Overview

### `face_addressed_mesh_2d.py`
Implements the `Mesh` class using the face-addressing scheme standard in industrial FVM codes. All topology and geometry are indexed by face. Each face stores:
- Two vertex indices
- Outward face area vector **S**_f (owner→neighbour for internal faces)
- Inverse cell-centre distance δ_f
- Linear interpolation weight f_x

Cell data (centre coordinates, area) are computed via the shoelace formula. The `generate_cartesian_mesh_files` function writes mesh files for an N_x × N_y structured Cartesian grid. The `Mesh.from_folder` class method reads these files back.

### `write_lid_driven_cavity_case.py`
Creates a complete case directory for the lid-driven cavity:
- Mesh files (via `generate_cartesian_mesh_files`)
- Initial fields `0/U.txt` and `0/p.txt` (zero everywhere)
- `params.json` storing `Re`, `nu`, `U_lid`, `d`, `rho`
- `bc.json` with velocity fixedValue BCs on all walls and zeroGradient pressure BCs; cell 0 is pinned to p = 0 via a `fixedCell` reference entry

### `sparse_cr.py`
- **`SparseCR`**: Compressed-Row sparse matrix assembled from triplet input; duplicate entries are summed. Supports matrix–vector products and diagonal extraction.
- **`LinearSystem`**: Wraps a `SparseCR` matrix and RHS vector. Provides a Gauss–Seidel smoother (`gauss_seidel`) and a tolerance-based iterative solve (`solve_gs_to_tol`) with optional residual history output.

### `fv_scalar.py`
Finite-volume scalar operators on a `Mesh`:
- **`gauss_grad_cell`**: Cell-centre gradient via Gauss theorem; fixedValue BCs override face values.
- **`assemble_laplace`**: Diffusion operator −∇·(γ∇ϕ) for uniform γ. fixedValue faces contribute implicitly to the diagonal and RHS; zeroGradient faces contribute nothing.
- **`assemble_convection_upwind`**: First-order upwind convection ∇·(uϕ) for a prescribed constant velocity field.
- **`assemble_ddt`**: Backward Euler time derivative term (for transient extensions).

### `momentum_predictor.py`
Solves the pressure-free steady momentum predictor:

∇·(uu) − ∇·(ν∇u) = 0

Separately for each velocity component (u_x, u_y). Convection is linearised with the face flux computed from the current cell-centre velocity field (frozen-coefficient approach). Implicit under-relaxation is applied before the GS solve, augmenting the diagonal by a_P → a_P/α_U and modifying the RHS accordingly. Returns the predicted velocity **u**★ and the pre-relaxation diagonals a_P^x and a_P^y for use in the pressure and correction steps.

### `pressure_correction.py`
Implements the pressure–velocity correction steps of SIMPLE:
- **`compute_face_flux_linear`**: Computes the precursor face flux F^pre = **S**_f · **u**_f using linear interpolation of **u**★.
- **`divergence_of_face_flux`**: Integrates the signed face flux over each cell to give ∇·F per cell.
- **`solve_pressure_equation`**: Assembles and solves the pressure-correction equation ∇·(1/a_P ∇p′) = ∇·**u**★ using variable-coefficient Laplacian assembly. The pressure diffusivity at a face is γ_f = ½(1/a_P^o + 1/a_P^n). All pressure BCs are zeroGradient (Neumann), so the system is singular; row 0 is replaced by p′_0 = 0. The solver is warm-started from the previous outer iteration.
- **`correct_face_flux`**: F^corr_f = F^pre_f − (1/a_P)_f |S_f| δ_f (p′_N − p′_P) for internal faces.
- **`correct_cell_velocity`**: **u**^corr = **u**★ − (1/a_P) ∇p′, using the Gauss gradient of p′.

### `simple_solver.py`
A self-contained SIMPLE outer loop with detailed diagnostic print statements (boundary flux checks, divergence checks). Intended for debugging and verification of individual algorithmic steps. On completion it writes `U_final.txt` and `p_final.txt` to the case directory. Convergence history files (`div_hist.txt`, `dU_hist.txt`) are **not** written by this script; use `Master_Analysis.py` or `re_variation_study.py` for runs that require history output.

---

## Running the Solver

### 1. Single-grid sanity check (20×20, Re=100)

```bash
python test20x20.py
```

Outputs go to `test_20x20_output/`. Runs for up to 400 outer SIMPLE iterations, terminating early if both convergence criteria are satisfied before the limit is reached. Produces convergence plots, centreline comparisons against Ghia (1982), pressure and velocity field plots, and matrix coefficient reports for Tasks 8 and 9.

### 2. Full multi-grid analysis (20×20, 40×40, 80×80 at Re=100)

```bash
python Master_Analysis.py
```

Outputs go to `analysis_output/`. Loads existing results if `U_final.txt` already exists in the case directory; otherwise runs the solver automatically. The 40×40 case is expected in `cavity_case/`; 20×20 and 80×80 use `cavity_20x20_Re100/` and `cavity_80x80_Re100/`.

### 3. Reynolds number study (Re=100/200/400/1000, 40×40)

```bash
python re_variation_study.py
```

Outputs go to `analysis_output/`. Loads from `cavity_40x40_Re{Re}/` if results exist; runs the solver otherwise. Under-relaxation factors are reduced progressively with Re as listed below.

### 4. Re-study post-processing

```bash
python re_variation_analysis.py
```

Outputs go to `re_analysis_output/`. Requires completed results in `cavity_40x40_Re{Re}/` for all four Re values.

---

## Configuration

Key parameters in `Master_Analysis.py`:

| Parameter    | Default    | Description                             |
|--------------|------------|-----------------------------------------|
| `RE`         | 100        | Reynolds number                         |
| `GRID_SIZES` | [20,40,80] | Grid sizes to run                       |
| `ALPHA_U`    | 0.7        | Momentum under-relaxation factor        |
| `GS_SWEEPS`  | 200        | Gauss–Seidel sweeps per linear solve    |
| `N_OUTER`    | 800        | Maximum SIMPLE outer iterations         |
| `TOL_DIV`    | 1e-5       | Convergence tolerance on ‖∇·F^corr‖_∞  |
| `TOL_DU`     | 1e-6       | Convergence tolerance on ‖ΔU‖_∞        |

## Case Directory Layout

Each case directory (e.g. `cavity_case/`) has the following structure after setup:

```
cavity_case/
├── mesh/
│   ├── points      # vertex coordinates
│   ├── faces       # vertex pairs per face
│   ├── cells       # face lists per cell
│   └── boundary    # patch name, start face, face count
├── 0/
│   ├── U.txt       # initial velocity field (nCells × 2)
│   └── p.txt       # initial pressure field (nCells,)
├── params.json     # Re, nu, U_lid, d, rho
└── bc.json         # velocity and pressure boundary conditions
```

After running the solver via `Master_Analysis.py`, `re_variation_study.py`, or `test20x20.py`, the following are written:

```
├── U_final.txt     # converged velocity field
├── p_final.txt     # converged pressure field
├── div_hist.txt    # ‖∇·F^corr‖_∞ per outer iteration
└── dU_hist.txt     # ‖ΔU‖_∞ per outer iteration
```

Note: `simple_solver.py` writes only `U_final.txt` and `p_final.txt`; it does not produce convergence history files.

---

## Dependencies

- Python ≥ 3.9
- NumPy
- Matplotlib

No external CFD libraries are used. All mesh, matrix, and solver components are implemented from scratch.

---

## Validation

Centreline velocity profiles are compared against the benchmark data of Ghia, Ghia, and Shin (1982) at Re = 100, 400, and 1000. The u-component centreline error converges at approximately first order (p ≈ 1.0) under mesh refinement from 20×20 to 80×80, consistent with the formal accuracy of the first-order upwind discretisation.

| Grid  | E_ℓ2(u) | E_ℓ∞(u) | E_ℓ2(v) | E_ℓ∞(v) |
|-------|---------|---------|---------|---------|
| 20×20 | 0.105   | 0.200   | 0.061   | 0.120   |
| 40×40 | 0.050   | 0.091   | 0.042   | 0.107   |
| 80×80 | 0.025   | 0.044   | 0.036   | 0.104   |

---

## Reference

U. Ghia, K.N. Ghia, C.T. Shin, *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method*, Journal of Computational Physics 48 (1982) 387–411.
