"""
Symmetry-aware RMSD calculation using graph isomorphism.

Used as a fallback when Hungarian+Kabsch RMSD is suspiciously large
despite similar energy/frequencies, which indicates symmetric equivalent
atoms (e.g. methyl groups in acetone) being mis-mapped.

Workflow:
  1. Hungarian+Kabsch -> rmsd_hung  (existing, fast)
  2. if rmsd_hung < threshold -> duplicate (done)
  3. if rmsd_hung >= threshold -> call symmetry_aware_rmsd():
     a. Build molecular graph from coordinates (covalent radii)
     b. Check graph isomorphism (NetworkX VF2)
        - Not isomorphic -> different structure (done)
        - Isomorphic -> enumerate all valid atom mappings
     c. For each mapping: Kabsch RMSD -> take minimum
     d. if min_rmsd < threshold -> duplicate (symmetry-induced false negative)

Dependencies: numpy, scipy, networkx
"""

import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

# Covalent radii (Angstrom) - Alvarez 2008
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28,
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71,
    'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
    'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Br': 1.20, 'I': 1.39,
}

def detect_bonds(elements, coords, scale=1.25):
    """Detect bonds from atomic coordinates using covalent radii.

    Args:
        elements: List of element symbols
        coords: Numpy array of shape (N, 3)
        scale: Tolerance factor for bond distance (default 1.25)

    Returns:
        List of (i, j) bond pairs
    """
    n = len(elements)
    bonds = []
    dists = cdist(coords, coords)
    for i in range(n):
        ri = COVALENT_RADII.get(elements[i], 1.5)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(elements[j], 1.5)
            if dists[i, j] < (ri + rj) * scale:
                bonds.append((i, j))
    return bonds

def _build_graph(elements, coords, bonds, heavy_only=True):
    """Build a NetworkX graph from molecular data.

    Args:
        elements: List of element symbols
        coords: Numpy array (N, 3)
        bonds: List of (i, j) bond pairs
        heavy_only: If True, only include non-H atoms

    Returns:
        (G, idx_map): NetworkX Graph and list mapping graph node -> original atom index
    """
    if heavy_only:
        idx_map = [i for i, e in enumerate(elements) if e != 'H']
    else:
        idx_map = list(range(len(elements)))
    idx_set = set(idx_map)
    # Map original index -> graph node index
    orig_to_node = {orig: node for node, orig in enumerate(idx_map)}

    G = nx.Graph()
    for node, orig in enumerate(idx_map):
        G.add_node(node, element=elements[orig])

    for (a, b) in bonds:
        if a in idx_set and b in idx_set:
            G.add_edge(orig_to_node[a], orig_to_node[b])

    return G, idx_map

def _kabsch_rmsd(P, Q):
    """Compute RMSD after optimal rotation (Kabsch algorithm).

    Args:
        P: Numpy array (N, 3) - reference coordinates
        Q: Numpy array (N, 3) - target coordinates

    Returns:
        RMSD value (float)
    """
    # Center both sets
    p_center = P.mean(axis=0)
    q_center = Q.mean(axis=0)
    p = P - p_center
    q = Q - q_center

    # Covariance matrix
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    q_rot = q @ R.T
    diff = p - q_rot
    return np.sqrt((diff * diff).sum() / len(P))

def check_graph_isomorphic(elements1, coords1, elements2, coords2,
                           bonds1=None, bonds2=None, heavy_only=True):
    """Check if two molecules are graph-isomorphic (same constitutional isomer).

    Lightweight check that only tests topology, no RMSD calculation.
    Used in energy safety net: graph isomorphic + dE ~ 0 -> same structure.

    Args:
        elements1, elements2: Lists of element symbols
        coords1, coords2: Numpy arrays of shape (N, 3)
        bonds1, bonds2: Pre-computed bond lists. If None, detected automatically.
        heavy_only: If True, compare heavy-atom graphs only

    Returns:
        bool: True if graphs are isomorphic (same constitution)
    """
    c1 = np.asarray(coords1, dtype=float)
    c2 = np.asarray(coords2, dtype=float)

    if bonds1 is None:
        bonds1 = detect_bonds(elements1, c1)
    if bonds2 is None:
        bonds2 = detect_bonds(elements2, c2)

    G1, idx1 = _build_graph(elements1, c1, bonds1, heavy_only=heavy_only)
    G2, idx2 = _build_graph(elements2, c2, bonds2, heavy_only=heavy_only)

    if len(idx1) != len(idx2):
        return False

    node_match = nx.algorithms.isomorphism.categorical_node_match('element', '')
    GM = GraphMatcher(G1, G2, node_match=node_match)
    return GM.is_isomorphic()

def symmetry_aware_rmsd(elements1, coords1, elements2, coords2,
                        bonds1=None, bonds2=None,
                        heavy_only=True, mirror=True,
                        max_mappings=1000):
    """Compute minimum RMSD considering molecular symmetry via graph isomorphism.

    This function enumerates all valid atom mappings (including symmetric
    equivalents like swapping two methyl groups) and returns the minimum
    Kabsch RMSD across all mappings.

    Args:
        elements1, elements2: Lists of element symbols
        coords1, coords2: Numpy arrays of shape (N, 3)
        bonds1, bonds2: Pre-computed bond lists. If None, detected automatically.
        heavy_only: If True, only use heavy atoms for RMSD
        mirror: If True, also check mirror image mappings
        max_mappings: Safety limit on number of mappings to evaluate

    Returns:
        float: Minimum RMSD across all valid mappings.
               Returns float('inf') if molecules are not graph-isomorphic.
    """
    c1 = np.asarray(coords1, dtype=float)
    c2 = np.asarray(coords2, dtype=float)

    # Detect bonds if not provided
    if bonds1 is None:
        bonds1 = detect_bonds(elements1, c1)
    if bonds2 is None:
        bonds2 = detect_bonds(elements2, c2)

    # Build molecular graphs
    G1, idx1 = _build_graph(elements1, c1, bonds1, heavy_only=heavy_only)
    G2, idx2 = _build_graph(elements2, c2, bonds2, heavy_only=heavy_only)

    # Quick check: same number of (heavy) atoms?
    if len(idx1) != len(idx2):
        return float('inf')

    # Graph isomorphism with element matching
    node_match = nx.algorithms.isomorphism.categorical_node_match('element', '')
    GM = GraphMatcher(G1, G2, node_match=node_match)

    if not GM.is_isomorphic():
        return float('inf')

    # Extract heavy atom coordinates
    hcoords1 = c1[idx1]
    hcoords2 = c2[idx2]

    # Enumerate all valid mappings and compute minimum RMSD
    min_rmsd = float('inf')
    n_mapped = 0
    for mapping in GM.isomorphisms_iter():
        # mapping: {G1_node -> G2_node}
        perm = [mapping[i] for i in range(len(idx1))]
        mapped_coords2 = hcoords2[perm]

        # Normal orientation
        rmsd = _kabsch_rmsd(hcoords1, mapped_coords2)
        min_rmsd = min(min_rmsd, rmsd)

        # Mirror image
        if mirror:
            mirrored = mapped_coords2.copy()
            mirrored[:, 0] *= -1  # Reflect x-axis
            rmsd_mir = _kabsch_rmsd(hcoords1, mirrored)
            min_rmsd = min(min_rmsd, rmsd_mir)

        n_mapped += 1
        if min_rmsd < 0.01:
            break  # Already found near-perfect match
        if n_mapped >= max_mappings:
            break  # Safety limit for highly symmetric molecules

    return min_rmsd

def compute_rmsd_with_symmetry_fallback(
    elements1, coords1, elements2, coords2,
    hungarian_rmsd, threshold,
    bonds1=None, bonds2=None,
    heavy_only=True, mirror=True
):
    """Main entry point: use existing Hungarian RMSD as primary,
    fall back to symmetry-aware RMSD only when needed.

    Args:
        elements1, elements2: Lists of element symbols
        coords1, coords2: Numpy arrays (N, 3)
        hungarian_rmsd: Pre-computed RMSD from Hungarian+Kabsch
        threshold: RMSD threshold for duplicate detection
        bonds1, bonds2: Pre-computed bond lists (optional)
        heavy_only: Use heavy atoms only
        mirror: Check mirror images

    Returns:
        (final_rmsd, method): Tuple of RMSD value and which method was used
            method is 'hungarian' or 'symmetry' or 'non_isomorphic'
    """
    # Case 1: Hungarian already says duplicate
    if hungarian_rmsd < threshold:
        return hungarian_rmsd, 'hungarian'

    # Case 2: Hungarian RMSD exceeds threshold -> try symmetry fallback
    sym_rmsd = symmetry_aware_rmsd(
        elements1, coords1, elements2, coords2,
        bonds1=bonds1, bonds2=bonds2,
        heavy_only=heavy_only, mirror=mirror
    )

    if sym_rmsd == float('inf'):
        return float('inf'), 'non_isomorphic'

    return sym_rmsd, 'symmetry'


# ============================================================
# Standalone test
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Test 1: Acetone - two equivalent methyl groups")
    print("=" * 60)

    elements = ['C', 'C', 'O', 'H', 'H', 'H', 'C', 'H', 'H', 'H']

    coords_A = np.array([
        [ 1.440,  0.046, -0.157],
        [-0.053,  0.019,  0.087],
        [-0.638,  0.965,  0.560],
        [ 1.652, -0.121, -1.216],
        [ 1.927, -0.763,  0.394],
        [ 1.852,  1.003,  0.154],
        [-0.774, -1.255, -0.295],
        [-0.358, -2.102,  0.257],
        [-1.836, -1.162, -0.080],
        [-0.629, -1.467, -1.357],
    ])

    coords_B = np.array([
        [-0.774, -1.255, -0.295],
        [-0.053,  0.019,  0.087],
        [-0.638,  0.965,  0.560],
        [-0.358, -2.102,  0.257],
        [-1.836, -1.162, -0.080],
        [-0.629, -1.467, -1.357],
        [ 1.440,  0.046, -0.157],
        [ 1.652, -0.121, -1.216],
        [ 1.927, -0.763,  0.394],
        [ 1.852,  1.003,  0.154],
    ])

    bonds_A = detect_bonds(elements, coords_A)
    bonds_B = detect_bonds(elements, coords_B)
    print(f"Bonds A: {bonds_A}")
    print(f"Bonds B: {bonds_B}")

    is_iso = check_graph_isomorphic(elements, coords_A, elements, coords_B,
                                    bonds1=bonds_A, bonds2=bonds_B)
    print(f"\nGraph isomorphic (heavy): {is_iso}")

    heavy_A = [i for i, e in enumerate(elements) if e != 'H']
    naive_rmsd = _kabsch_rmsd(coords_A[heavy_A], coords_B[heavy_A])
    print(f"Naive same-index RMSD (heavy): {naive_rmsd:.4f} A")

    sym_rmsd = symmetry_aware_rmsd(
        elements, coords_A, elements, coords_B,
        bonds1=bonds_A, bonds2=bonds_B,
        heavy_only=True, mirror=True
    )
    print(f"Symmetry-aware RMSD (heavy):   {sym_rmsd:.4f} A")

    final_rmsd, method = compute_rmsd_with_symmetry_fallback(
        elements, coords_A, elements, coords_B,
        hungarian_rmsd=naive_rmsd, threshold=0.3,
        bonds1=bonds_A, bonds2=bonds_B,
        heavy_only=True, mirror=True
    )
    print(f"\nFinal RMSD: {final_rmsd:.4f} A (method: {method})")
    print(f"Duplicate (threshold 0.3)? {'YES' if final_rmsd < 0.3 else 'NO'}")

    print("\n" + "=" * 60)
    print("Test 2: Acetone vs Propanal")
    print("=" * 60)

    elements_prop = ['C', 'C', 'O', 'H', 'H', 'H', 'C', 'H', 'H', 'H']
    coords_prop = np.array([
        [-1.280,  0.260,  0.000],
        [-0.010, -0.560,  0.000],
        [ 1.090,  0.300,  0.000],
        [ 1.100,  1.500,  0.000],
        [-1.280,  0.880,  0.880],
        [-1.280,  0.880, -0.880],
        [-2.170, -0.370,  0.000],
        [-0.010, -1.180,  0.880],
        [-0.010, -1.180, -0.880],
        [ 2.020, -0.260,  0.000],
    ])

    is_iso2 = check_graph_isomorphic(elements, coords_A, elements_prop, coords_prop)
    print(f"Graph isomorphic: {is_iso2}")
    if not is_iso2:
        print("Result: Different constitutional isomers (correct!)")