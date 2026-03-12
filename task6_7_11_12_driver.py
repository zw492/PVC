from __future__ import annotations

import os
import io
import json
import contextlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from face_addressed_mesh_2d import Mesh
from write_lid_driven_cavity_case import write_lid_driven_cavity_case
from momentum_predictor import momentum_predictor
from pressure_correction import (
    compute_face_flux_linear,
    solve_pressure_equation,
    correct_face_flux,
    correct_cell_velocity,
    divergence_of_face_flux,
)

# -----------------------------------------------------------------------------
# 这里固定成你报告正文的基线算例（如 40x40, Re=100）
# -----------------------------------------------------------------------------
NX = 40
NY = 40
RE = 100
ALPHA_U = 0.7
GS_SWEEPS = 200
N_OUTER_MAX = 1200
TOL_DIV = 1e-13
TOL_DU = None      # 如果想再加一个速度变化停止准则，可改成浮点数
CASE_DIR = f"taskdiag_case_{NX}x{NY}_Re{RE}"
OUT_DIR = Path(f"taskdiag_output_{NX}x{NY}_Re{RE}")
OUT_DIR.mkdir(exist_ok=True)

# Task 7 （40x40）
# 生成代表性单元
# 你想和某张现成截图完全一致，可把这里改成对应编号。
PREFERRED_SAMPLE_CELLS = [8, 303, 600, 792, 1215]


def choose_sample_cells(n_cells: int) -> list[int]:
    """选取 Task 7 展示用的单元编号。
    - 预设编号在网格范围内，就先用
    - 网格较小（20x20），就自动过滤非法编号，必要时均匀
    """
    out = [c for c in PREFERRED_SAMPLE_CELLS if 0 <= c < n_cells]
    if len(out) >= 5:
        return out[:5]
    fallback = np.linspace(0, n_cells - 1, 5, dtype=int).tolist()
    out = []
    for c in fallback:
        if c not in out:
            out.append(c)
    return out


def ensure_case() -> None:
    """算例不在，就自动生成"""
    if not os.path.exists(CASE_DIR):
        write_lid_driven_cavity_case(
            case_dir=CASE_DIR, nx=NX, ny=NY, d=0.1, U_lid=1.0, Re=RE
        )


def load_case():
    """读网格、初始场、物性参数和边界条件。"""
    mesh = Mesh.from_folder(f"{CASE_DIR}/mesh")
    bc = json.loads(Path(f"{CASE_DIR}/bc.json").read_text())
    params = json.loads(Path(f"{CASE_DIR}/params.json").read_text())
    nu = float(params["nu"])
    bcU = bc["U"]
    bcp = bc.get("p", None)
    if bcp:
        bcp = {k: v for k, v in bcp.items() if k in {pt.name for pt in mesh.patches}}
    U = np.loadtxt(f"{CASE_DIR}/0/U.txt")
    p = np.loadtxt(f"{CASE_DIR}/0/p.txt")
    return mesh, U, p, nu, bcU, bcp


def save_csv(path: Path, header: str, arr: np.ndarray) -> None:
    """统一保存 CSV"""
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    print(f"[Saved] {path}")


def plot_task12(inner_histories: dict, outer_iters: np.ndarray, div_hist: np.ndarray, dU_hist: np.ndarray) -> None:
    """Task 12 图：内层残差 + 外层收敛。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 动量方程inner GS
    ax = axes[0, 0]
    if "Ux_iter1" in inner_histories:
        k, r = inner_histories["Ux_iter1"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.8, label="Ux, outer iter 1")
    if "Ux_iter_last" in inner_histories:
        k, r = inner_histories["Ux_iter_last"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.8, ls="--", label="Ux, final outer iter")
    if "Uy_iter1" in inner_histories:
        k, r = inner_histories["Uy_iter1"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.2, label="Uy, outer iter 1")
    if "Uy_iter_last" in inner_histories:
        k, r = inner_histories["Uy_iter_last"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.2, ls=":", label="Uy, final outer iter")
    ax.set_title("Task 12 — inner GS residuals (momentum predictor)")
    ax.set_xlabel("GS sweep")
    ax.set_ylabel("Residual norm")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    # 压力修正方程inner GS
    ax = axes[0, 1]
    if "p_iter1" in inner_histories:
        k, r = inner_histories["p_iter1"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.8, label="p', outer iter 1")
    if "p_iter_last" in inner_histories:
        k, r = inner_histories["p_iter_last"]
        ax.semilogy(k, np.clip(r, 1e-30, None), lw=1.8, ls="--", label="p', final outer iter")
    ax.set_title("Task 12 — inner GS residuals (pressure correction)")
    ax.set_xlabel("GS sweep")
    ax.set_ylabel("Residual norm")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    # 外层 div(Fcorr)
    ax = axes[1, 0]
    ax.semilogy(outer_iters, np.clip(div_hist, 1e-30, None), lw=1.8)
    ax.axhline(TOL_DIV, color="red", ls=":", lw=1.0, label=f"TOL_DIV={TOL_DIV:.0e}")
    ax.set_title("Task 12 — outer convergence (divergence residual)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\|\nabla\cdot F_{corr}\|_\infty$")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    # 外层速度变化量收敛
    ax = axes[1, 1]
    ax.semilogy(outer_iters, np.clip(dU_hist, 1e-30, None), lw=1.8)
    ax.set_title("Task 12 — outer convergence (velocity change)")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\|\Delta U\|_\infty$")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "task12_inner_outer_convergence.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out}")


def main() -> None:
    ensure_case()
    mesh, U, p, nu, bcU, bcp = load_case()
    n_cells = len(mesh.cells)

    # 所有边界 face：Task 6 要检查 F_pre 在这些 face 上是否为 0
    boundary_faces = sorted({f for patch in mesh.patches for f in patch.face_ids})
    sample_cells = choose_sample_cells(n_cells)

    # 历史量：分别用于 Task 6 / Task 11 / Task 12
    div_hist: list[float] = []
    dU_hist: list[float] = []
    task6_rows: list[list[float]] = []
    task11_rows: list[list[float]] = []
    p_prime_init = np.zeros(n_cells)   # pressure correction warm start

    # 保存代表性inner residual 曲线
    inner_histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    final_task7_rows: list[list[float]] = []
    stop_reason = "max_outer"

    for it in range(1, N_OUTER_MAX + 1):
        U_old = U.copy()

        # 动量预测抓取inner GS residual history
        with contextlib.redirect_stdout(io.StringIO()):
            Ustar, aP_x, aP_y, mom_hist = momentum_predictor(
                mesh, U, nu, bcU, alphaU=ALPHA_U, gs_sweeps=GS_SWEEPS, return_history=True
            )

        Fpre = compute_face_flux_linear(mesh, Ustar, bcU)

        # 压力修正抓取 inner GS residual history
        with contextlib.redirect_stdout(io.StringIO()):
            p_prime, p_hist = solve_pressure_equation(
                mesh,
                Fpre,
                aP_x,
                aP_y,
                bc_p=bcp,
                ref_cell=0,
                gs_sweeps=GS_SWEEPS,
                p_init=p_prime_init,
                return_history=True,
                tol_abs=1e-12,
                tol_rel=1e-10,
                report_every=2000,
            )
        p_prime_init = p_prime.copy()

        # 得到修正后的通量、速度与压力
        Fcorr = correct_face_flux(mesh, Fpre, p_prime, aP_x, aP_y)
        U = correct_cell_velocity(mesh, Ustar, p_prime, aP_x, aP_y, bc_p=bcp)
        p += p_prime

        div_pre = divergence_of_face_flux(mesh, Fpre)
        div_corr = divergence_of_face_flux(mesh, Fcorr)

        div_inf = float(np.max(np.abs(div_corr)))
        dU_inf = float(np.max(np.abs(U - U_old)))
        div_hist.append(div_inf)
        dU_hist.append(dU_inf)

        # Task 6：这些量理论上应为 0（数值上到 roundoff/solver tolerance）
        task6_rows.append([
            it,
            float(np.max(np.abs(Fpre[boundary_faces]))),
            float(np.sum(Fpre[boundary_faces])),
            float(np.sum(div_pre)),
        ])

        # Task 11：inner solve初始/最终 residual
        ux_hist = np.array(mom_hist["Ux"], dtype=float)
        uy_hist = np.array(mom_hist["Uy"], dtype=float)
        p_hist_arr = np.array(p_hist, dtype=float)
        task11_rows.append([
            it,
            ux_hist[0, 1], ux_hist[-1, 1],
            uy_hist[0, 1], uy_hist[-1, 1],
            p_hist_arr[0, 1], p_hist_arr[-1, 1],
        ])

        # Task 12：开始和最后的inner 曲线
        if it == 1:
            inner_histories["Ux_iter1"] = (ux_hist[:, 0], ux_hist[:, 1])
            inner_histories["Uy_iter1"] = (uy_hist[:, 0], uy_hist[:, 1])
            inner_histories["p_iter1"] = (p_hist_arr[:, 0], p_hist_arr[:, 1])

        if it % 25 == 0 or it == 1:
            print(f"iter {it:4d}: div_inf={div_inf:.3e}, dU_inf={dU_inf:.3e}")

        stop_div = div_inf < TOL_DIV
        stop_du = True if TOL_DU is None else (dU_inf < TOL_DU)
        if stop_div and stop_du:
            stop_reason = "tolerance"
            print(f"[Stop] outer iter {it}: div_inf={div_inf:.3e}, dU_inf={dU_inf:.3e}")
            break

    # 记录最终一次 inner 曲线
    outer_final = len(div_hist)
    inner_histories["Ux_iter_last"] = (ux_hist[:, 0], ux_hist[:, 1])
    inner_histories["Uy_iter_last"] = (uy_hist[:, 0], uy_hist[:, 1])
    inner_histories["p_iter_last"] = (p_hist_arr[:, 0], p_hist_arr[:, 1])

    # Task 7：在最终迭代抽取若干代表性单元，比较 net Fpre 与 net Fcorr
    for c in sample_cells:
        final_task7_rows.append([
            outer_final,
            c,
            float(div_pre[c]),
            float(div_corr[c]),
        ])

    # 保存CSV
    save_csv(
        OUT_DIR / "task6_zero_checks.csv",
        "iteration,max_abs_boundary_Fpre,sum_boundary_Fpre,global_sum_cell_flux_pre",
        np.array(task6_rows, dtype=float),
    )
    save_csv(
        OUT_DIR / "task7_cell_flux_table.csv",
        "iteration,cell_number,net_flux_pre,net_flux_corr",
        np.array(final_task7_rows, dtype=float),
    )
    save_csv(
        OUT_DIR / "task11_inner_solver_residuals.csv",
        "iteration,r0_Ux,rf_Ux,r0_Uy,rf_Uy,r0_p,rf_p",
        np.array(task11_rows, dtype=float),
    )
    save_csv(
        OUT_DIR / "task12_outer_convergence.csv",
        "iteration,div_inf,dU_inf",
        np.column_stack([np.arange(1, outer_final + 1), np.array(div_hist), np.array(dU_hist)]),
    )

    # 代表性inner residual 曲线存 CSV
    for key, (k, r) in inner_histories.items():
        save_csv(OUT_DIR / f"{key}.csv", "sweep,residual", np.column_stack([k, r]))

    # Task 12 图
    plot_task12(inner_histories, np.arange(1, outer_final + 1), np.array(div_hist), np.array(dU_hist))

    # 摘要
    summary = [
        "PVC task diagnostics summary",
        "=" * 60,
        f"Grid: {NX}x{NY}, Re={RE}, alpha_U={ALPHA_U}",
        f"GS sweeps per inner solve: {GS_SWEEPS}",
        f"Outer iterations executed: {outer_final}",
        f"Stopping reason: {stop_reason}",
        f"Final div_inf: {div_hist[-1]:.6e}",
        f"Final dU_inf: {dU_hist[-1]:.6e}",
        "",
        "Task 6 (numerical verification):",
        f"  max |F_pre| on boundary faces over all iterations = {max(row[1] for row in task6_rows):.6e}",
        f"  max |sum(boundary F_pre)| over all iterations     = {max(abs(row[2]) for row in task6_rows):.6e}",
        f"  max |global sum over all cell fluxes|            = {max(abs(row[3]) for row in task6_rows):.6e}",
        "",
        "Task 7 (representative cells at final iteration):",
    ]
    for row in final_task7_rows:
        summary.append(
            f"  iter={int(row[0])}, cell={int(row[1])}: net_flux_pre={row[2]:.6e}, net_flux_corr={row[3]:.6e}"
        )
    summary.append("")
    summary.append("Task 11 CSV: task11_inner_solver_residuals.csv")
    summary.append("Task 12 plot: task12_inner_outer_convergence.png")
    (OUT_DIR / "task_diagnostics_summary.txt").write_text("\n".join(summary))
    print(f"[Saved] {OUT_DIR / 'task_diagnostics_summary.txt'}")


if __name__ == "__main__":
    main()
