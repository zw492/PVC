# Tasks 1-2
# face_addressed_mesh_2d.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import List, Tuple, Dict, Iterable, Optional

Vec2 = Tuple[float, float]

# Parsing utilities
_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)

def _strip_comments(s: str) -> str:
    return re.sub(_COMMENT_RE, "", s)

def _tokenize(s: str) -> List[str]:
    """
    Tokenize a foam-ish ASCII file containing:
      - integers/floats
      - identifiers (patch names)
      - parentheses
    """
    s = _strip_comments(s)
    # Keep parentheses as tokens
    # Words / numbers / parens
    return re.findall(r"[A-Za-z_]\w*|[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+|\(|\)", s)

class _TokStream:
    def __init__(self, toks: List[str]):
        self.toks = toks
        self.i = 0

    def peek(self) -> str:
        if self.i >= len(self.toks):
            raise ValueError("Unexpected end of tokens")
        return self.toks[self.i]

    def pop(self) -> str:
        t = self.peek()
        self.i += 1
        return t

    def expect(self, t: str) -> None:
        got = self.pop()
        if got != t:
            raise ValueError(f"Expected token '{t}', got '{got}'")

def _read_int(ts: _TokStream) -> int:
    return int(ts.pop())

def _read_float(ts: _TokStream) -> float:
    return float(ts.pop())

def _read_list_of_points(ts: _TokStream, n: int) -> List[Vec2]:
    """
    Accepts either (x y) or (x y z); ignores z.
    """
    pts: List[Vec2] = []
    ts.expect("(")
    for _ in range(n):
        ts.expect("(")
        x = _read_float(ts)
        y = _read_float(ts)
        # optional z
        if ts.peek() != ")":
            _ = _read_float(ts)
        ts.expect(")")
        pts.append((x, y))
    ts.expect(")")
    return pts

def _read_faces(ts: _TokStream, n: int) -> List[List[int]]:
    """
    Each face is: k( i0 i1 ... ik-1 )
    """
    faces: List[List[int]] = []
    ts.expect("(")
    for _ in range(n):
        k = _read_int(ts)
        ts.expect("(")
        verts = [_read_int(ts) for _ in range(k)]
        ts.expect(")")
        faces.append(verts)
    ts.expect(")")
    return faces

def _read_cells(ts: _TokStream, n: int) -> List[List[int]]:
    """
    Each cell is: k( f0 f1 ... fk-1 ) where f* are face indices
    """
    cells: List[List[int]] = []
    ts.expect("(")
    for _ in range(n):
        k = _read_int(ts)
        ts.expect("(")
        fids = [_read_int(ts) for _ in range(k)]
        ts.expect(")")
        cells.append(fids)
    ts.expect(")")
    return cells

def _read_boundary(ts: _TokStream) -> List[Tuple[str, List[int]]]:
    """
    boundary file format:
      nPatches
      (
        patchName
        nFaces
        (
          f0 f1 ...
        )
        patchName2
        ...
      )
    """
    patches: List[Tuple[str, List[int]]] = []
    ts.expect("(")
    while ts.peek() != ")":
        name = ts.pop()
        nFaces = _read_int(ts)
        ts.expect("(")
        fids = [_read_int(ts) for _ in range(nFaces)]
        ts.expect(")")
        patches.append((name, fids))
    ts.expect(")")
    return patches

# Geometry helpers (2D)
def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])

def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])

def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]

def _scale(s: float, a: Vec2) -> Vec2:
    return (s * a[0], s * a[1])

def _norm(a: Vec2) -> float:
    return math.hypot(a[0], a[1])

def _mean(points: List[Vec2]) -> Vec2:
    inv = 1.0 / len(points)
    x = sum(p[0] for p in points) * inv
    y = sum(p[1] for p in points) * inv
    return (x, y)

def _order_polygon_vertices(vertices: List[Vec2]) -> List[Vec2]:
    """
    Given an unordered set of vertices of a convex-ish polygon (Cartesian cell),
    return them ordered CCW around centroid.
    """
    c = _mean(vertices)
    def ang(p: Vec2) -> float:
        return math.atan2(p[1] - c[1], p[0] - c[0])
    return sorted(vertices, key=ang)

def _polygon_area_and_centroid_ccw(poly: List[Vec2]) -> Tuple[float, Vec2]:
    """
    Shoelace for area; centroid formula for simple polygon.
    Assumes CCW order (but works with CW area sign).
    """
    n = len(poly)
    if n < 3:
        return 0.0, _mean(poly)

    A2 = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A2 += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    if abs(A2) < 1e-14:
        return 0.0, _mean(poly)

    A = 0.5 * A2
    C = (Cx / (3.0 * A2), Cy / (3.0 * A2))
    return abs(A), C

def _face_center(points: List[Vec2], face_verts: List[int]) -> Vec2:
    return _mean([points[i] for i in face_verts])

def _face_Sf_raw(points: List[Vec2], face_verts: List[int]) -> Vec2:
    """
    2D face area vector: normal * length.
    For Cartesian, each face is a segment with 2 vertices.
    If >2 verts, we approximate using first and last (still okay for our generated mesh).
    Orientation is based on vertex order.
    """
    if len(face_verts) < 2:
        raise ValueError("Face must have at least 2 vertices in 2D.")
    p0 = points[face_verts[0]]
    p1 = points[face_verts[1]]
    t = _sub(p1, p0)
    # Rotate +90 degrees: (tx,ty) -> (ty,-tx)
    return (t[1], -t[0])

# Mesh data structures
@dataclass
class BoundaryPatch:
    name: str
    face_ids: List[int]

@dataclass
class Mesh:
    points: List[Vec2]                  # point coordinates
    faces: List[List[int]]              # per face: vertex indices
    cells: List[List[int]]              # per cell: face indices
    patches: List[BoundaryPatch]        # boundary patch list

    # Derived:
    face_owner: List[int]               # per face: owner cell
    face_neighbour: List[int]           # per face: neighbour cell or -1
    cell_neighbours: List[List[int]]    # per cell: neighbouring cells (unique)
    cell_centers: List[Vec2]
    cell_areas: List[float]
    face_centers: List[Vec2]
    Sf: List[Vec2]                      # oriented from owner -> neighbour (or outward for boundary)
    magSf: List[float]
    delta: List[float]
    fx: List[float]

    def patch_owner_cells(self) -> Dict[str, List[int]]:
        """
        Return mapping patchName -> list of owner cell indices adjacent to that patch.
        """
        out: Dict[str, List[int]] = {}
        for p in self.patches:
            cells = [self.face_owner[f] for f in p.face_ids]
            # unique but keep stable ordering
            seen = set()
            uniq = []
            for c in cells:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            out[p.name] = uniq
        return out

    @staticmethod
    def from_folder(folder: str | Path) -> "Mesh":
        folder = Path(folder)
        pts = _parse_points(folder / "points")
        faces = _parse_faces(folder / "faces")
        cells = _parse_cells(folder / "cells")
        patches = _parse_boundary(folder / "boundary")

        mesh = _build_mesh(pts, faces, cells, patches)
        return mesh

# File parsers
def _parse_points(path: Path) -> List[Vec2]:
    toks = _tokenize(path.read_text())
    ts = _TokStream(toks)
    n = _read_int(ts)
    return _read_list_of_points(ts, n)

def _parse_faces(path: Path) -> List[List[int]]:
    toks = _tokenize(path.read_text())
    ts = _TokStream(toks)
    n = _read_int(ts)
    return _read_faces(ts, n)

def _parse_cells(path: Path) -> List[List[int]]:
    toks = _tokenize(path.read_text())
    ts = _TokStream(toks)
    n = _read_int(ts)
    return _read_cells(ts, n)

def _parse_boundary(path: Path) -> List[BoundaryPatch]:
    toks = _tokenize(path.read_text())
    ts = _TokStream(toks)
    nP = _read_int(ts)
    patches_raw = _read_boundary(ts)
    if len(patches_raw) != nP:
        # tolerate mismatch; still store what we read
        pass
    return [BoundaryPatch(name, fids) for name, fids in patches_raw]

# Build derived topology + geometry
def _build_mesh(
    points: List[Vec2],
    faces: List[List[int]],
    cells: List[List[int]],
    patches: List[BoundaryPatch],
) -> Mesh:
    nF = len(faces)
    nC = len(cells)

    # 1) Face -> adjacent cells list from cells' face lists
    face_to_cells: List[List[int]] = [[] for _ in range(nF)]
    for c, fids in enumerate(cells):
        for f in fids:
            face_to_cells[f].append(c)

    # 2) Compute provisional cell centers/areas (needed to orient Sf and owner/neighbour)
    cell_centers: List[Vec2] = []
    cell_areas: List[float] = []
    for fids in cells:
        # Collect unique vertex indices from all faces of the cell
        vid_set = set()
        for f in fids:
            for v in faces[f]:
                vid_set.add(v)
        vids = sorted(vid_set)
        verts = [points[v] for v in vids]
        ordered = _order_polygon_vertices(verts)
        area, cen = _polygon_area_and_centroid_ccw(ordered)
        cell_centers.append(cen)
        cell_areas.append(area)
    
    # A2: For 2D Cartesian use, enforce each face is a 2-point segment
    for f, fv in enumerate(faces):
        if len(fv) != 2:
            raise ValueError(
                f"2D Cartesian mesh expects 2-vertex faces; face {f} has {len(fv)} vertices."
            )

    # 3) Face centers + raw Sf
    face_centers: List[Vec2] = [_face_center(points, fv) for fv in faces]
    Sf_raw: List[Vec2] = [_face_Sf_raw(points, fv) for fv in faces]

    # 4) Owner / neighbour and oriented Sf
    face_owner = [-1] * nF
    face_neighbour = [-1] * nF
    Sf: List[Vec2] = [(0.0, 0.0)] * nF

    for f in range(nF):
        adj = face_to_cells[f]
        if len(adj) == 0:
            raise ValueError(f"Face {f} is not referenced by any cell.")
        elif len(adj) == 1:
            c0 = adj[0]
            face_owner[f] = c0
            face_neighbour[f] = -1
            # Orient outward: ensure Sf points from cell centre to outside (towards face centre)
            v = _sub(face_centers[f], cell_centers[c0])
            s = Sf_raw[f]
            if _dot(s, v) < 0.0:
                s = _scale(-1.0, s)
            Sf[f] = s
        elif len(adj) == 2:
            cA, cB = adj[0], adj[1]
            dAB = _sub(cell_centers[cB], cell_centers[cA])
            s = Sf_raw[f]
            # If s points from A to B, dot(s,dAB) > 0
            if _dot(s, dAB) >= 0.0:
                face_owner[f] = cA
                face_neighbour[f] = cB
                Sf[f] = s
            else:
                face_owner[f] = cB
                face_neighbour[f] = cA
                Sf[f] = _scale(-1.0, s)
        else:
            raise ValueError(f"Non-manifold face {f} shared by {len(adj)} cells (expected 1 or 2).")
    
    # A1: Boundary patch completeness check (must be AFTER face_neighbour is computed)
    boundary_faces = [f for f in range(nF) if face_neighbour[f] == -1]

    # build face -> patch count
    face_patch_count = [0] * nF
    for p in patches:
        for f in p.face_ids:
            face_patch_count[f] += 1

    # patched faces must actually be boundary faces
    for p in patches:
        for f in p.face_ids:
            if face_neighbour[f] != -1:
                raise ValueError(f"Patch '{p.name}' contains internal face {f}")

    # every boundary face must appear in exactly one patch
    missing = [f for f in boundary_faces if face_patch_count[f] == 0]
    if missing:
        raise ValueError(f"Some boundary faces are not in any patch (first 20): {missing[:20]}")

    dup = [f for f in boundary_faces if face_patch_count[f] > 1]
    if dup:
        raise ValueError(f"Some boundary faces appear in multiple patches (first 20): {dup[:20]}")


    # 5) Cell neighbours list from internal faces
    cell_neigh: List[set[int]] = [set() for _ in range(nC)]
    for f in range(nF):
        o = face_owner[f]
        n = face_neighbour[f]
        if n != -1:
            cell_neigh[o].add(n)
            cell_neigh[n].add(o)
    cell_neighbours = [sorted(list(s)) for s in cell_neigh]

    # 6) Discretisation support: magSf, delta, fx
    magSf: List[float] = [0.0] * nF
    delta: List[float] = [0.0] * nF
    fx: List[float] = [1.0] * nF  # default boundary=1

    for f in range(nF):
        o = face_owner[f]
        n = face_neighbour[f]

        # |Sf|
        magSf[f] = math.hypot(Sf[f][0], Sf[f][1])

        Cf = face_centers[f]
        Co = cell_centers[o]

        if n != -1:
            Cn = cell_centers[n]
            PN = (Cn[0] - Co[0], Cn[1] - Co[1])
            dPN = math.hypot(PN[0], PN[1])
            if dPN < 1e-14:
                raise ValueError(f"Face {f}: zero PN distance")
            delta[f] = 1.0 / dPN

            # fx = |fN|/|PN| = |Cf - Cn| / |Cn - Co|
            fN = (Cf[0] - Cn[0], Cf[1] - Cn[1])
            dfN = math.hypot(fN[0], fN[1])
            fx[f] = dfN / dPN
        else:
            # boundary: delta = 1/|Pf| where Pf = Cf - Co
            Pf = (Cf[0] - Co[0], Cf[1] - Co[1])
            dPf = math.hypot(Pf[0], Pf[1])
            if dPf < 1e-14:
                raise ValueError(f"Face {f}: zero Pf distance")
            delta[f] = 1.0 / dPf
            fx[f] = 1.0

    return Mesh(
        points=points,
        faces=faces,
        cells=cells,
        patches=patches,
        face_owner=face_owner,
        face_neighbour=face_neighbour,
        cell_neighbours=cell_neighbours,
        cell_centers=cell_centers,
        cell_areas=cell_areas,
        face_centers=face_centers,
        Sf=Sf,
        magSf=magSf,
        delta=delta,
        fx=fx,
    )


# Cartesian mesh generator (PVC Task 2 helper)
def generate_cartesian_mesh_files(
    out_folder: str | Path,
    nx: int,
    ny: int,
    Lx: float = 1.0,
    Ly: float = 1.0,
    patch_names: Tuple[str, str, str, str] = ("left", "right", "bottom", "top"),
) -> None:
    """
    Writes points/faces/cells/boundary in the same simple ASCII style.
    Faces are 2-vertex segments. Cells are quads described by 4 face indices.

    patch_names: (left, right, bottom, top)
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Points indexing: (i,j) -> pid
    def pid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    dx = Lx / nx
    dy = Ly / ny

    points: List[Vec2] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            points.append((i * dx, j * dy))

    faces: List[List[int]] = []
    # Keep a mapping from edge (p0,p1) undirected to face id
    edge_to_fid: Dict[Tuple[int, int], int] = {}

    def add_face(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_to_fid:
            return edge_to_fid[key]
        fid = len(faces)
        # Store with a->b orientation (will be re-oriented later by geometry anyway)
        faces.append([a, b])
        edge_to_fid[key] = fid
        return fid

    cells: List[List[int]] = []
    # Boundary patch face lists
    left_faces: List[int] = []
    right_faces: List[int] = []
    bottom_faces: List[int] = []
    top_faces: List[int] = []

    for j in range(ny):
        for i in range(nx):
            p00 = pid(i, j)
            p10 = pid(i + 1, j)
            p11 = pid(i + 1, j + 1)
            p01 = pid(i, j + 1)

            # Faces (edges) of quad: bottom, right, top, left
            f_bottom = add_face(p00, p10)
            f_right  = add_face(p10, p11)
            f_top    = add_face(p11, p01)
            f_left   = add_face(p01, p00)

            cells.append([f_bottom, f_right, f_top, f_left])

            # Boundary collection
            if i == 0:
                left_faces.append(f_left)
            if i == nx - 1:
                right_faces.append(f_right)
            if j == 0:
                bottom_faces.append(f_bottom)
            if j == ny - 1:
                top_faces.append(f_top)

    patches = [
        (patch_names[0], sorted(set(left_faces))),
        (patch_names[1], sorted(set(right_faces))),
        (patch_names[2], sorted(set(bottom_faces))),
        (patch_names[3], sorted(set(top_faces))),
    ]

    # Write files
    _write_points(out_folder / "points", points)
    _write_faces(out_folder / "faces", faces)
    _write_cells(out_folder / "cells", cells)
    _write_boundary(out_folder / "boundary", patches)

def _write_points(path: Path, points: List[Vec2]) -> None:
    lines = [str(len(points)), "(",]
    for x, y in points:
        lines.append(f"({x:g} {y:g})")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")

def _write_faces(path: Path, faces: List[List[int]]) -> None:
    lines = [str(len(faces)), "(",]
    for verts in faces:
        lines.append(f"{len(verts)}({ ' '.join(str(v) for v in verts) })")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")

def _write_cells(path: Path, cells: List[List[int]]) -> None:
    lines = [str(len(cells)), "(",]
    for fids in cells:
        lines.append(f"{len(fids)}({ ' '.join(str(f) for f in fids) })")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")

def _write_boundary(path: Path, patches: List[Tuple[str, List[int]]]) -> None:
    lines = [str(len(patches)), "("]
    for name, fids in patches:
        lines.append(name)
        lines.append(str(len(fids)))
        lines.append("(")
        lines.append(" ".join(str(f) for f in fids))
        lines.append(")")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n")

# Quick self-test
if __name__ == "__main__":
    # Generate a 4x3 mesh and read it back
    folder = Path("mesh_demo_4x3")
    generate_cartesian_mesh_files(folder, nx=4, ny=3, Lx=1.0, Ly=1.0)

    m = Mesh.from_folder(folder)
    print("nPoints, nFaces, nCells, nPatches:",
          len(m.points), len(m.faces), len(m.cells), len(m.patches))
    print("Cell 0 center/area:", m.cell_centers[0], m.cell_areas[0])
    print("Face 0 owner/neigh, Sf:", m.face_owner[0], m.face_neighbour[0], m.Sf[0])
    print("Neighbours of cell 0:", m.cell_neighbours[0])
    print("Sample face fx/delta/magSf for first 5 faces:")
    for f in range(5):
        print(f"f={f}: neigh={m.face_neighbour[f]}, fx={m.fx[f]:.4f}, delta={m.delta[f]:.4f}, |Sf|={m.magSf[f]:.4f}")

