from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tflux.pipeline.config as config
from tflux.utils.logging import get_logger
from tflux.dtypes import Junction
import numpy as np
import pandas as pd

logger = get_logger(__name__)

# ============================================================
# OBJ loader (triangles only)
# ============================================================

# Defaults to column 0 = t, 1 = y, 2 = x from our data
def load_obj_tri_mesh(path: str) -> Dict[str, Any]:
    vs: List[Tuple[float, float, float]] = []
    faces_v: List[Tuple[int, int, int]] = []

    def parse_face_token(tok: str) -> int:
        # token forms: "v", "v//vn", "v/vt", "v/vt/vn"
        parts = tok.split("/")
        return int(parts[0]) - 1

    p = Path(path)
    if p.exists() and p.is_dir():
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
                _, t, y, x = line.split()[:4]
                if config.do_scaling:
                    t = float(t) * config.dt
                    y = float(y) * config.dx
                    x = float(x) * config.dx
                    vs.append((float(t), float(y), float(x)))
                else:
                    vs.append((float(t), float(y), float(x)))
            elif line.startswith("f "):
                toks = line.split()[1:]
                if len(toks) != 3:
                    raise ValueError(f"Non-triangular face encountered: {line}")
                a = parse_face_token(toks[0])
                b = parse_face_token(toks[1])
                c = parse_face_token(toks[2])
                faces_v.append((a, b, c))

    V = np.asarray(vs, dtype=np.float64)
    Fv = np.asarray(faces_v, dtype=np.int64)
    logger.debug(f"Loaded OBJ mesh: {len(V)} vertices, {len(Fv)} faces")
    return {"vertices": V, "faces_v": Fv}


# ============================================================
# Geometry / adjacency
# ============================================================
def face_normals_from_geometry(vertices: np.ndarray, faces_v: np.ndarray) -> np.ndarray:
    v0 = vertices[faces_v[:, 0]]
    v1 = vertices[faces_v[:, 1]]
    v2 = vertices[faces_v[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-12)


def build_face_adjacency_and_pairs(faces_v: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    logger.info("Building face adjacency and edge pairs")
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
    adj: List[List[int]],
    *,
    normal_weight: float = 1.0,
    geom_weight: float = 0.5,
) -> np.ndarray:
    """
    Build a per-face feature vector that combines:
      - Unit normal            (3 dims)  — orientation
      - Normalised centroid    (3 dims)  — spatial position
      - Mean neighbour dihedral angle (1 dim) — local curvature
      - Log face area          (1 dim)  — shape scale

    All feature groups are individually z-score standardised before
    weighting so that no single group dominates by raw magnitude.
    The returned matrix is L2-row-normalised and ready for k-means.

    Args:
        normal_weight: scalar multiplier applied to the normal sub-block.
        geom_weight:   scalar multiplier applied to centroid / curvature /
                       area sub-block.
    Returns:
        features: (F, 8) float64, each row L2-normalised.
    """
    F = faces_v.shape[0]

    # --- (a) unit normals (already unit length) ---
    normals = face_n.copy()  # (F, 3)

    # --- (b) face centroids, z-score standardised ---
    v0 = vertices[faces_v[:, 0]]
    v1 = vertices[faces_v[:, 1]]
    v2 = vertices[faces_v[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)
    c_mean = centroids.mean(axis=0)
    c_std = centroids.std(axis=0) + 1e-12
    centroids_norm = (centroids - c_mean) / c_std  # (F, 3)

    # --- (c) mean dihedral angle to face-graph neighbours ---
    # dot product of neighbouring normals → arccos → mean per face
    neigh_dots = np.zeros(F, dtype=np.float64)
    for i in range(F):
        nb = adj[i]
        if nb:
            dots = np.clip(face_n[nb] @ face_n[i], -1.0, 1.0)
            neigh_dots[i] = np.arccos(dots).mean()
        # faces with no neighbours keep 0
    nd_mean = neigh_dots.mean()
    nd_std = neigh_dots.std() + 1e-12
    curvature = (neigh_dots - nd_mean) / nd_std  # (F,)

    # --- (d) log face area, z-score standardised ---
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # (F,)
    log_area = np.log(areas + 1e-12)
    la_mean = log_area.mean()
    la_std = log_area.std() + 1e-12
    log_area_norm = (log_area - la_mean) / la_std  # (F,)

    # --- assemble and weight ---
    geom_block = np.column_stack([centroids_norm, curvature, log_area_norm])  # (F, 5)
    feat = np.concatenate(
        [normal_weight * normals, geom_weight * geom_block], axis=1
    )  # (F, 8)

    # L2-row-normalise so k-means distance is well-conditioned
    row_norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.maximum(row_norms, 1e-12)

    logger.info(
        f"Computed geometry features: shape={feat.shape}, "
        f"normal_weight={normal_weight}, geom_weight={geom_weight}"
    )

    # Log df.head(5)
    logger.debug(f"Feature array shape: {feat.shape}")
    return feat


# ============================================================
# K-means on arbitrary feature vectors (Euclidean)
# ============================================================
def kmeans_euclidean(
    X: np.ndarray,
    k: int,
    n_iter: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard Euclidean k-means with k-means++-style initialisation.

    Args:
        X:      (F, D) feature matrix (rows need not be unit vectors).
        k:      number of clusters.
        n_iter: maximum iterations.
        seed:   RNG seed.
    Returns:
        labels:  (F,) int64
        centers: (k, D) float64
    """
    rng = np.random.default_rng(seed)
    F = X.shape[0]
    logger.info(f"Running Euclidean k-means on {F} feature vectors (dim={X.shape[1]}) with k={k}, n_iter={n_iter}, seed={seed}")

    # k-means++ init: first center random, subsequent chosen with
    # probability proportional to squared distance from nearest center
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(0, F)]
    for ci in range(1, k):
        dists = np.array([
            np.min(np.sum((X - centers[cj]) ** 2, axis=1))
            for cj in range(ci)
        ])  # (F,) squared distances to nearest existing center
        # farthest-point seeding (deterministic; use probabilistic if preferred)
        centers[ci] = X[int(np.argmax(dists))]

    labels = np.zeros(F, dtype=np.int64)
    for iteration in range(n_iter):
        # assignment: squared Euclidean distance to each center
        # computed as ||x||^2 - 2 x·c + ||c||^2  (broadcast)
        sq_x = (X ** 2).sum(axis=1, keepdims=True)           # (F,1)
        sq_c = (centers ** 2).sum(axis=1)                     # (k,)
        cross = X @ centers.T                                 # (F,k)
        dists2 = sq_x - 2.0 * cross + sq_c                   # (F,k)
        new_labels = np.argmin(dists2, axis=1).astype(np.int64)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update centers
        for ci in range(k):
            mask = labels == ci
            if not np.any(mask):
                centers[ci] = X[rng.integers(0, F)]
            else:
                centers[ci] = X[mask].mean(axis=0)

    logger.debug(f"Euclidean k-means completed in {iteration + 1} iterations")
    return labels, centers


# ============================================================
# K-means on unit normals (cosine distance)
# ============================================================
def kmeans_unit_vectors_cosine(X: np.ndarray, k: int, n_iter: int = 30, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (F,3) unit vectors
    Returns:
      labels: (F,)
      centers: (k,3) unit vectors
    Cosine distance ~ (1 - dot).
    """
    rng = np.random.default_rng(seed)
    F = X.shape[0]

    # kmeans++-ish init: pick 1 random, then farthest by cosine
    logger.info(f"Running k-means on {F} unit normals with k={k}, n_iter={n_iter}, seed={seed}")
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = X[rng.integers(0, F)]
    for ci in range(1, k):
        dots = X @ centers[:ci].T                      # (F,ci)
        best = np.max(dots, axis=1)                    # closest center by cosine (max dot)
        dist = 1.0 - best
        idx = int(np.argmax(dist))
        centers[ci] = X[idx]

    # iterate
    labels = np.zeros(F, dtype=np.int64)
    for _ in range(n_iter):
        dots = X @ centers.T           # (F,k)
        new_labels = np.argmax(dots, axis=1)  # max cosine similarity
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update centers: mean then renormalize
        for ci in range(k):
            mask = labels == ci
            if not np.any(mask):
                centers[ci] = X[rng.integers(0, F)]
                continue
            m = X[mask].mean(axis=0)
            centers[ci] = m / (np.linalg.norm(m) + 1e-12)
    logger.info(f"K-means completed in {_+1} iterations")
    return labels, centers


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
) -> List[Junction]:
    """
    Partition the entire mesh into k coherent segments (no holes):
      1) Build per-face feature vectors combining unit normals with local
         geometry descriptors (centroid, mean-neighbour dihedral angle,
         log face area).  The balance between orientation and geometry is
         controlled by *normal_weight* and *geom_weight*.
      2) Euclidean k-means on the combined (L2-normalised) feature vectors.
      3) Smooth labels via ICM on the adjacency graph (uses face normals only,
         so the MRF unary / pairwise terms remain interpretable).
      4) Remove small islands.
      5) Build Junction per label (unique vertices).

    Args:
        normal_weight: Weight applied to the 3-dim unit-normal sub-block
            before L2-row-normalisation.  Higher values push clustering to
            favour orientation over spatial / curvature features.
        geom_weight:   Weight applied to the 5-dim geometry sub-block
            (normalised centroid x3, mean dihedral, log area).  Increase to
            make spatial position and local curvature more influential.

    Returns exactly k Junctions unless some label becomes empty (rare; can
    happen on degenerate meshes).
    """
    obj_path = Path(obj_path)
    mesh = load_obj_tri_mesh(str(obj_path))
    V = mesh["vertices"]
    Fv = mesh["faces_v"]

    face_n = face_normals_from_geometry(V, Fv)
    adj, pairs = build_face_adjacency_and_pairs(Fv)

    # 1) Build combined feature vectors and cluster
    logger.info(f"Building geometry features with normal_weight={normal_weight}, geom_weight={geom_weight}")
    feat = compute_face_geometry_features(
        V, Fv, face_n, adj,
        normal_weight=normal_weight,
        geom_weight=geom_weight,
    )
    labels, _feat_centers = kmeans_euclidean(feat, k=k, n_iter=kmeans_iter, seed=seed)

    # 2) smooth labels on the mesh graph (fills holes, enforces coherence)
    logger.info(f"Smoothing labels with ICM: n_iter={smooth_iter}, lam={lam}")
    labels, centers = smooth_labels_icm(labels, face_n, adj, k=k, n_iter=smooth_iter, lam=lam)

    # 3) remove small islands (topology cleanup)
    logger.info(f"Removing small components with fewer than {min_island_faces} faces")
    labels = relabel_small_components(labels, adj, min_faces=min_island_faces, k=k)

    # 4) build Junctions per label
    logger.info(f"Building Junctions from k-means labels")
    junctions: List[Junction] = []
    for roi_index in range(k):
        faces_in = np.where(labels == roi_index)[0]
        if faces_in.size == 0:
            continue
        verts_idx = np.unique(Fv[faces_in].reshape(-1))
        verts = V[verts_idx]

        j = Junction(vertices=verts, roi_index=roi_index)
        j.source_file = obj_path
        junctions.append(j)

    # keep deterministic order (0..k-1)
    junctions.sort(key=lambda jj: jj.roi_index)
    logger.info(f"Extracted {len(junctions)} junctions from {obj_path} with k={k}")
    return junctions
