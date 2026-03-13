from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tflux.pipeline.config as config
from tflux.utils.logging import get_logger
from tflux.dtypes import Junction, Cell
import numpy as np
import pandas as pd
from codetiming import Timer

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False

logger = get_logger(__name__)

def get_xp(prefer_gpu: bool = True):
    """Return cupy if available and preferred, else numpy."""
    return cp if (CUPY_AVAILABLE and prefer_gpu) else np

# ============================================================
# OBJ loader (triangles only)
# ============================================================

# Defaults to column 0 = t, 1 = y, 2 = x from our data
def load_obj_tri_mesh(path: str) -> Dict[str, Any]:
    vs: List[Tuple[float, float, float]] = []
    vn: List[Tuple[float, float, float]] = []
    faces_v: List[Tuple[int, int, int]] = []
    faces_vn: List[Tuple[int, int, int]] = []

    def parse_face_token(tok: str) -> Tuple[int, int]:
        """Parse OBJ face token. Supports: 'v', 'v//vn', 'v/vt/vn'"""
        parts = tok.split("/")   # split on single '/'
        v_idx = int(parts[0]) - 1
        vn_idx = int(parts[2]) - 1 if len(parts) >= 3 and parts[2] else v_idx
        return v_idx, vn_idx

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OBJ file not found: {p}")
    if p.is_dir():
        raise ValueError(f"Expected a .obj file path, got directory: {p}")
    
    if config.do_scaling:
        logger.info(f"Loading OBJ with scaling to SI units: dt={config.dt:.4f}, dx={config.dx:.4f}")
    else:
        logger.info(f"Loading OBJ without scaling to SI units: dt={config.dt:.4f}, dx={config.dx:.4f} (scaling disabled)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    logger.warning(f"Skipping malformed vertex line: {line!r}")
                    continue
                _, t, y, x = parts[:4]
                if config.do_scaling:
                    vs.append((float(t) * config.dt, float(y) * config.dx, float(x) * config.dx))
                else:
                    vs.append((float(t), float(y), float(x)))
            elif line.startswith("vn "):
                _, tn, yn, xn = line.split()[:4]
                vn.append((float(tn), float(yn), float(xn)))
            elif line.startswith("f "):
                toks = line.split()[1:]  # List of strings ["60//60", "70//70", "80//80"]
                if len(toks) != 3:
                    raise ValueError(f"Non-triangular face encountered: {line}")
                a, an = parse_face_token(toks[0])
                b, bn = parse_face_token(toks[1])
                c, cn = parse_face_token(toks[2])
                faces_v.append((a, b, c))
                faces_vn.append((an, bn, cn))

    V = np.asarray(vs, dtype=np.float64)
    Vn = np.asarray(vn, dtype=np.float64)
    Fv = np.asarray(faces_v, dtype=np.int64)
    Fvn = np.asarray(faces_vn, dtype=np.int64)
    logger.debug(f"Loaded OBJ mesh: {len(V)} vertices, {len(Fvn)} face norms, and {len(Fv)} faces")
    return {"vertices": V, "norms": Vn, "faces_v": Fv, "faces_vn": Fvn}


# ============================================================
# Geometry / adjacency
# ============================================================
def face_normals_from_geometry(vertices: np.ndarray, faces_v: np.ndarray) -> np.ndarray:
    xp = get_xp()
    with Timer(text="face_normals_from_geometry: {:.3f}s", logger=logger.debug):
        vertices_g = xp.asarray(vertices)
        faces_v_g  = xp.asarray(faces_v)
        v0 = vertices_g[faces_v_g[:, 0]]
        v1 = vertices_g[faces_v_g[:, 1]]
        v2 = vertices_g[faces_v_g[:, 2]]
        n = xp.cross(v1 - v0, v2 - v0)
        norm = xp.linalg.norm(n, axis=1, keepdims=True)
        Fv = n / xp.maximum(norm, 1e-12)
        if xp is cp:
            cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(Fv) if xp is cp else Fv


def face_normals_from_label(norms: np.ndarray, faces_vn: np.ndarray) -> np.ndarray:
    xp = get_xp()
    with Timer(text="face_normals_from_label: {:.3f}s", logger=logger.debug):
        norms_g    = xp.asarray(norms)
        faces_vn_g = xp.asarray(faces_vn)
        n0 = norms_g[faces_vn_g[:, 0]]
        n1 = norms_g[faces_vn_g[:, 1]]
        n2 = norms_g[faces_vn_g[:, 2]]
        n = (n0 + n1 + n2) / 3
        norm = xp.linalg.norm(n, axis=1, keepdims=True)
        Fvn = n / xp.maximum(norm, 1e-12)
        if xp is cp:
            cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(Fvn) if xp is cp else Fvn


def build_face_adjacency_and_pairs(faces_v: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    F = faces_v.shape[0]
    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}

    for fi in range(F):
        a, b, c = faces_v[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(e, []).append(fi)

    adj = [[] for _ in range(F)]
    pairs: List[Tuple[int, int]] = []
    for faces in edge_to_faces.values():
        if len(faces) < 2:
            continue
        faces_sorted = sorted(faces)
        for ii in range(len(faces_sorted)):
            for jj in range(ii + 1, len(faces_sorted)):
                f1, f2 = faces_sorted[ii], faces_sorted[jj]
                adj[f1].append(f2)
                adj[f2].append(f1)
                pairs.append((f1, f2))

    adj = [sorted(set(nbrs)) for nbrs in adj]
    logger.debug(f"Built adjacency for {F} faces with {len(pairs)} adjacent face pairs")
    return adj, np.asarray(pairs, dtype=np.int64)


def dihedral_angles(face_n: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    dots = np.einsum("ij,ij->i", face_n[pairs[:, 0]], face_n[pairs[:, 1]])
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


# ============================================================
# Local geometry feature extraction
# ============================================================
def compute_face_geometry_features(
    vertices: np.ndarray,
    faces_v: np.ndarray,
    face_n: np.ndarray,
    *,
    normal_weight: float = 1.0,
    geom_weight: float = 0.5,
) -> np.ndarray:
    xp = get_xp()
    F = faces_v.shape[0]

    with Timer(text="  feature centroids (GPU): {:.3f}s", logger=logger.debug):
        vertices_g = xp.asarray(vertices)
        faces_v_g  = xp.asarray(faces_v)
        face_n_g   = xp.asarray(face_n)
        normals    = face_n_g.copy()
        v0 = vertices_g[faces_v_g[:, 0]]
        v1 = vertices_g[faces_v_g[:, 1]]
        v2 = vertices_g[faces_v_g[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        c_mean = centroids.mean(axis=0)
        c_std  = centroids.std(axis=0) + 1e-12
        centroids_norm = (centroids - c_mean) / c_std
        if xp is cp:
            cp.cuda.Stream.null.synchronize()

    with Timer(text="  feature assemble+normalise (GPU): {:.3f}s", logger=logger.debug):
        feat = xp.concatenate(
            [normal_weight * normals, geom_weight * centroids_norm], axis=1
        )
        row_norms = xp.linalg.norm(feat, axis=1, keepdims=True)
        feat = feat / xp.maximum(row_norms, 1e-12)
        if xp is cp:
            cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(feat) if xp is cp else feat


# ============================================================
# K-means on arbitrary feature vectors (Euclidean)
# ============================================================

def kmeans_euclidean(
    X: np.ndarray,
    k: int,
    n_iter: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    xp = get_xp()
    rng = np.random.default_rng(seed)
    F = X.shape[0]

    with Timer(text="  kmeans init: {:.3f}s", logger=logger.debug):
        Xg = xp.asarray(X, dtype=xp.float64)
        centers = xp.empty((k, Xg.shape[1]), dtype=xp.float64)
        centers[0] = Xg[rng.integers(0, F)]
        min_dists = xp.sum((Xg - centers[0]) ** 2, axis=1)
        for ci in range(1, k):
            centers[ci] = Xg[int(xp.argmax(min_dists))]
            new_dists = xp.sum((Xg - centers[ci]) ** 2, axis=1)
            xp.minimum(min_dists, new_dists, out=min_dists)
        t_mid = (xp.max(Xg[:, 0]) + xp.min(Xg[:, 0])) / 2
        centers[:, 0] = t_mid

    sq_x   = (Xg ** 2).sum(axis=1, keepdims=True)
    labels = xp.full(F, -1, dtype=xp.int64)
    iteration = 0

    with Timer(text="  kmeans iterations: {:.3f}s", logger=logger.debug):
        for iteration in range(n_iter):
            sq_c   = (centers ** 2).sum(axis=1)
            cross  = Xg @ centers.T
            dists2 = sq_x - 2.0 * cross + sq_c
            new_labels = xp.argmin(dists2, axis=1).astype(xp.int64)
            if xp.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                centers[ci] = Xg[mask].mean(axis=0) if xp.any(mask) else Xg[rng.integers(0, F)]
            centers[:, 0] = t_mid
        if xp is cp:
            cp.cuda.Stream.null.synchronize()

    logger.debug(f"Euclidean k-means completed in {iteration + 1} iterations")
    labels_out  = cp.asnumpy(labels)  if xp is cp else labels
    centers_out = cp.asnumpy(centers) if xp is cp else centers
    return labels_out, centers_out


# ============================================================
# Smooth labels on the face graph (ICM on an MRF)
# ============================================================
def smooth_labels_icm(
    labels: np.ndarray,
    face_n: np.ndarray,
    adj: List[List[int]],
    k: int,
    *,
    n_iter: int = 10,
    lam: float = 0.8,
    dihedral_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterated Conditional Modes:
      For each face i, choose label l minimizing:
        unary(i,l) + lam * sum_{j in N(i)} w_ij * [l != label(j)]

    unary(i,l) = 1 - dot(n_i, center_l)
    centers updated each outer iteration (mean normals per label).

    dihedral_weight (optional): array of per-edge weights aligned to adj edges is hard;
    instead we pass None and use uniform weights, or pass a function-style weight.
    For simplicity, we do uniform weights here.
    """
    F = labels.shape[0]

    def compute_centers(lbls: np.ndarray) -> np.ndarray:
        centers = np.zeros((k, 3), dtype=np.float64)
        for l in range(k):
            m = face_n[lbls == l].mean(axis=0) if np.any(lbls == l) else np.array([0.0, 0.0, 1.0])
            centers[l] = m / (np.linalg.norm(m) + 1e-12)
        return centers

    centers = compute_centers(labels)

    for _ in range(n_iter):
        centers = compute_centers(labels)
        changed = 0

        # precompute unary: (F,k) = 1 - dot(n, center)
        unary = 1.0 - (face_n @ centers.T)

        for i in range(F):
            neigh = adj[i]
            if not neigh:
                continue

            # pairwise cost for choosing each label l:
            # lam * (#neighbors with different label)
            neigh_labels = labels[neigh]
            # counts per label among neighbors
            counts = np.bincount(neigh_labels, minlength=k)
            # if choose label l, mismatches = deg - counts[l]
            deg = len(neigh)
            pair_cost = lam * (deg - counts)

            new_l = int(np.argmin(unary[i] + pair_cost))
            if new_l != labels[i]:
                labels[i] = new_l
                changed += 1

        if changed == 0:
            break

    centers = compute_centers(labels)
    return labels, centers


# ============================================================
# Remove tiny islands (enforce local topology)
# ============================================================
def relabel_small_components(
    labels: np.ndarray,
    adj: List[List[int]],
    *,
    min_faces: int = 200,
    k: Optional[int] = None,
) -> np.ndarray:
    """
    For each label, find connected components (on face graph).
    Any component smaller than min_faces is reassigned to a neighboring majority label.
    """
    F = labels.shape[0]
    if k is None:
        k = int(labels.max()) + 1

    visited = np.zeros(F, dtype=bool)

    for start in range(F):
        if visited[start]:
            continue
        lab = int(labels[start])
        # BFS for this component under same label
        stack = [start]
        visited[start] = True
        comp = []
        boundary_nbr_labels = []

        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in adj[f]:
                if labels[nb] == lab and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
                elif labels[nb] != lab:
                    boundary_nbr_labels.append(int(labels[nb]))

        if len(comp) >= min_faces:
            continue
        if not boundary_nbr_labels:
            continue

        # reassign to most common neighboring label
        new_lab = int(np.bincount(np.asarray(boundary_nbr_labels), minlength=k).argmax())
        labels[np.asarray(comp, dtype=np.int64)] = new_lab

    return labels


# ============================================================
# Partition mesh into k segments, then return Junctions
# ============================================================
def extract_junctions(
    obj_path: str | Path,
    *,
    k: int = 3,
    kmeans_iter: int = 40,
    smooth_iter: int = 4,
    lam: float = 0.9,
    min_island_faces: int = 400,
    seed: int = 0,
    normal_weight: float = 1.0,
    geom_weight: float = 0.5,
    cell: Cell
) -> List[Junction]:
    obj_path = Path(obj_path)

    with Timer(text="[1/6] load_obj_tri_mesh: {:.3f}s", logger=logger.info):
        mesh = load_obj_tri_mesh(str(obj_path))
    V   = mesh["vertices"]
    Vn  = mesh["norms"]
    Fv  = mesh["faces_v"]
    Fvn = mesh["faces_vn"]

    with Timer(text="[2/6] face_normals: {:.3f}s", logger=logger.info):
        face_n_geom  = face_normals_from_geometry(V, Fv)
        face_n_label = face_normals_from_label(Vn, Fvn)

    cell.vertices    = V
    cell.norms       = Vn
    cell.face_n_geom  = face_n_geom
    cell.face_n_label = face_n_label

    with Timer(text="[3/6] Building face adjacency and edge pairs: {:.3f}s", logger=logger.info):
        adj, pairs = build_face_adjacency_and_pairs(Fv)

    with Timer(text="[4/6] compute_face_geometry_features: {:.3f}s", logger=logger.info):
        feat = compute_face_geometry_features(
            V, Fv, face_n_label,
            normal_weight=normal_weight,
            geom_weight=geom_weight,
        )

    with Timer(text="[5/6] Running Euclidean k-means: {:.3f}s", logger=logger.info):
        labels, _feat_centers = kmeans_euclidean(feat, k=k, n_iter=kmeans_iter, seed=seed)

    with Timer(text="[6/6] smooth_labels_icm: {:.3f}s", logger=logger.info):
        labels, centers = smooth_labels_icm(labels, face_n_label, adj, k=k, n_iter=smooth_iter, lam=lam)

    with Timer(text="[+] relabel_small_components: {:.3f}s", logger=logger.info):
        labels = relabel_small_components(labels, adj, min_faces=min_island_faces, k=k)

    logger.info(f"Building Junctions from k-means labels")
    junctions: List[Junction] = []
    for roi_index in range(k):
        faces_in  = np.where(labels == roi_index)[0]
        if faces_in.size == 0:
            continue
        verts_idx = np.unique(Fv[faces_in].reshape(-1))
        verts     = V[verts_idx]
        j = Junction(vertices=verts, roi_index=roi_index)
        j.source_file = obj_path
        junctions.append(j)

    junctions.sort(key=lambda jj: jj.roi_index)
    cell.junctions = junctions
    logger.info(f"Extracted {len(junctions)} junctions from {obj_path} with k={k}")
    return junctions
