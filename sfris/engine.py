#!/usr/bin/env python3
"""Molecular data types, topology analysis, fragmentation, and clustering."""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sfris.io import (
    SFRISParams, COVALENT_RADII,
    BOND_THRESHOLD, SINGLE_BOND_MIN_RATIO, DOUBLE_BOND_MIN_RATIO,
    DEFAULT_GAP_FRACTIONS,
)

# ============================================================
# Atom / Bond / Molecule
# ============================================================
@dataclass
class Atom:
    """Single atom with element and coordinates."""
    element: str
    x: float
    y: float
    z: float
    index: int = 0
    original_index: int = -1

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: 'Atom') -> float:
        return np.linalg.norm(self.coords - other.coords)

@dataclass
class Bond:
    """Bond between two atoms."""
    atom1: int
    atom2: int
    order: int = 1
    in_ring: bool = False
    cuttable: bool = True
    priority: float = 1.0

@dataclass
class Molecule:
    """Molecule with atoms and bonds."""
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    energy: Optional[float] = None

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]

    @property
    def coords(self) -> np.ndarray:
        return np.array([[a.x, a.y, a.z] for a in self.atoms])

    @property
    def formula(self) -> str:
        counts = {}
        for atom in self.atoms:
            counts[atom.element] = counts.get(atom.element, 0) + 1
        result = ""
        for elem in ['C', 'H']:
            if elem in counts:
                result += elem + (str(counts[elem]) if counts[elem] > 1 else "")
                del counts[elem]
        for elem in sorted(counts.keys()):
            result += elem + (str(counts[elem]) if counts[elem] > 1 else "")
        return result

    def to_xyz(self, comment: str = "") -> str:
        lines = [str(self.n_atoms), comment]
        for atom in self.atoms:
            lines.append(
                f"{atom.element:2s} {atom.x:15.10f} {atom.y:15.10f} {atom.z:15.10f}"
            )
        return "\n".join(lines)

    def save_xyz(self, filepath: str, comment: str = ""):
        with open(filepath, 'w') as f:
            f.write(self.to_xyz(comment) + "\n")

    def copy(self) -> 'Molecule':
        return Molecule(
            atoms=[Atom(a.element, a.x, a.y, a.z, a.index, a.original_index)
                   for a in self.atoms],
            bonds=[Bond(b.atom1, b.atom2, b.order, b.in_ring, b.cuttable,
                        b.priority) for b in self.bonds],
            charge=self.charge,
            multiplicity=self.multiplicity,
            energy=self.energy,
        )

    def sort_by_original_index(self):
        if not self.atoms or self.atoms[0].original_index < 0:
            return
        self.atoms.sort(key=lambda a: a.original_index)
        for i, atom in enumerate(self.atoms):
            atom.index = i

# ============================================================
# Topology Analyzer
# ============================================================
class TopologyAnalyzer:
    """Analyze molecular topology: bonds, rings, priorities."""

    def __init__(self, mol: Molecule, params: SFRISParams = None):
        self.mol = mol
        self.params = params
        self.adjacency: Dict[int, List[int]] = {}

    def analyze(self) -> Molecule:
        self._detect_bonds()
        self._detect_rings()
        self._assign_priorities()
        self._apply_cut_restrictions()
        return self.mol

    def _detect_bonds(self):
        self.mol.bonds = []
        self.adjacency = {i: [] for i in range(self.mol.n_atoms)}
        for i, atom1 in enumerate(self.mol.atoms):
            for j, atom2 in enumerate(self.mol.atoms):
                if j <= i:
                    continue
                dist = atom1.distance_to(atom2)
                r1 = COVALENT_RADII.get(atom1.element, 1.5)
                r2 = COVALENT_RADII.get(atom2.element, 1.5)
                sum_radii = r1 + r2
                if dist < sum_radii * BOND_THRESHOLD:
                    ratio = dist / sum_radii
                    if ratio >= SINGLE_BOND_MIN_RATIO:
                        order = 1
                    elif ratio >= DOUBLE_BOND_MIN_RATIO:
                        order = 2
                    else:
                        order = 3
                    bond = Bond(atom1=i, atom2=j, order=order)
                    self.mol.bonds.append(bond)
                    self.adjacency[i].append(j)
                    self.adjacency[j].append(i)

    def _detect_rings(self):
        visited = set()
        ring_bonds = set()

        def dfs(node: int, parent: int, path: List[int]):
            visited.add(node)
            path.append(node)
            for neighbor in self.adjacency[node]:
                if neighbor == parent:
                    continue
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    for k in range(len(cycle)):
                        a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                        ring_bonds.add((min(a, b), max(a, b)))
                elif neighbor not in visited:
                    dfs(neighbor, node, path)
            path.pop()

        for i in range(self.mol.n_atoms):
            if i not in visited:
                dfs(i, -1, [])

        for bond in self.mol.bonds:
            key = (min(bond.atom1, bond.atom2), max(bond.atom1, bond.atom2))
            bond.in_ring = key in ring_bonds

    def _assign_priorities(self):
        for bond in self.mol.bonds:
            atom1 = self.mol.atoms[bond.atom1]
            atom2 = self.mol.atoms[bond.atom2]
            elem1, elem2 = atom1.element, atom2.element
            elems = {elem1, elem2}

            bond.cuttable = True
            bond.priority = 0.5

            # C-H bonds are never cut
            if elems == {'C', 'H'}:
                bond.cuttable = False
                bond.priority = 0.0
                continue

            # Aromatic-like ring bonds with multiple bond character
            if bond.in_ring and bond.order > 1:
                bond.priority = 0.05
                continue

            if 'H' in elems:
                other = elem1 if elem2 == 'H' else elem2
                if other in ('O', 'N'):
                    bond.priority = 1.0
                elif other in ('S', 'Se'):
                    bond.priority = 0.9
                elif other == 'P':
                    bond.priority = 0.8
                elif other in ('B', 'Si'):
                    bond.priority = 0.6
                else:
                    bond.priority = 0.4
                continue

            # Triple bonds
            if bond.order == 3:
                if elems == {'C'}:
                    bond.priority = 0.1
                elif 'N' in elems:
                    bond.priority = 0.1
                continue

            # Double bonds
            if bond.order == 2:
                if 'O' in elems:
                    bond.priority = 0.4
                elif 'N' in elems:
                    bond.priority = 0.4
                elif elems == {'C'}:
                    bond.priority = 0.3
                else:
                    bond.priority = 0.3
                continue

            # Single bonds involving carbon
            if 'C' in elems:
                other = elem1 if elem2 == 'C' else elem2
                if other in ('O', 'N'):
                    bond.priority = 0.9
                elif other in ('S', 'Se'):
                    bond.priority = 0.8
                elif other in ('F', 'Cl', 'Br', 'I'):
                    bond.priority = 0.8
                elif other in ('P', 'B', 'Si'):
                    bond.priority = 0.7
                elif other == 'C':
                    if bond.in_ring:
                        bond.priority = 0.5
                    else:
                        neighbors1 = self.adjacency.get(bond.atom1, [])
                        neighbors2 = self.adjacency.get(bond.atom2, [])
                        has_hetero = False
                        for n in neighbors1 + neighbors2:
                            if self.mol.atoms[n].element not in ('C', 'H'):
                                has_hetero = True
                                break
                        bond.priority = 0.8 if has_hetero else 0.6
                continue

            # Heteroatom-heteroatom
            if 'C' not in elems and 'H' not in elems:
                if bond.order == 3:
                    bond.priority = 0.1
                elif bond.order == 2:
                    bond.priority = 0.3
                elif bond.in_ring:
                    bond.priority = 0.5
                else:
                    bond.priority = 0.7
                continue

            # Ring bonds default
            if bond.in_ring:
                bond.priority = 0.5

    def _apply_cut_restrictions(self):
        if self.params is None:
            return
        for bond in self.mol.bonds:
            if not bond.cuttable:
                continue
            if bond.order > 1 and not self.params.cut_multiple_bond:
                bond.cuttable = False
                bond.priority = 0.0
                continue
            if bond.in_ring and not self.params.cut_ring:
                bond.cuttable = False
                bond.priority = 0.0
                continue
            if not self.params.cut_functional_group:
                atom1 = self.mol.atoms[bond.atom1]
                atom2 = self.mol.atoms[bond.atom2]
                elems = {atom1.element, atom2.element}
                has_heteroatom = any(e not in ('C', 'H') for e in elems)
                if has_heteroatom:
                    bond.cuttable = False
                    bond.priority = 0.0

    def get_cuttable_bonds(self, min_priority: float = 0.0) -> List[Bond]:
        return [b for b in self.mol.bonds
                if b.cuttable and b.priority >= min_priority]

# ============================================================
# Fragmenter
# ============================================================
class Fragmenter:
    """Handle molecule fragmentation and recombination."""

    def __init__(self, params: SFRISParams):
        self.params = params
        self.rng = np.random.default_rng()
        self.fibonacci_directions = self._generate_fibonacci_directions(
            params.placements_per_cut
        )
        self.placement_attempts = params.placement_attempts

    @staticmethod
    def _generate_fibonacci_directions(n: int) -> List[np.ndarray]:
        if n <= 0:
            return []
        if n == 1:
            return [np.array([1.0, 0.0, 0.0])]
        directions = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1.0 - (i / (n - 1)) * 2.0
            radius = np.sqrt(1.0 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            directions.append(np.array([x, y, z]))
        return directions

    def enumerate_cut_combinations(self, mol: Molecule) -> List[List[Bond]]:
        cuttable = [b for b in mol.bonds
                    if b.cuttable and b.priority >= self.params.min_priority]
        if not cuttable:
            return []
        all_combinations = []
        max_cuts = min(self.params.max_cuts, len(cuttable))
        for n_cuts in range(1, max_cuts + 1):
            for combo in combinations(cuttable, n_cuts):
                all_combinations.append(list(combo))
        return all_combinations

    def cut_molecule(self, mol: Molecule, bonds_to_cut: List[Bond]) -> List[Molecule]:
        if not bonds_to_cut:
            return [mol.copy()]

        cut_pairs = set()
        for b in bonds_to_cut:
            cut_pairs.add((b.atom1, b.atom2))
            cut_pairs.add((b.atom2, b.atom1))

        adjacency = {i: [] for i in range(mol.n_atoms)}
        for bond in mol.bonds:
            if (bond.atom1, bond.atom2) not in cut_pairs:
                adjacency[bond.atom1].append(bond.atom2)
                adjacency[bond.atom2].append(bond.atom1)

        # BFS to find connected components
        visited = set()
        fragments = []

        def bfs(start: int) -> List[int]:
            component = []
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return component

        for i in range(mol.n_atoms):
            if i not in visited:
                fragments.append(bfs(i))

        # Build fragment Molecule objects
        fragment_mols = []
        for component in fragments:
            frag = Molecule(charge=0, multiplicity=1)
            index_map = {}
            for new_idx, old_idx in enumerate(component):
                old_atom = mol.atoms[old_idx]
                orig_idx = (old_atom.original_index
                            if old_atom.original_index >= 0 else old_idx)
                new_atom = Atom(old_atom.element, old_atom.x, old_atom.y,
                                old_atom.z, new_idx, orig_idx)
                frag.atoms.append(new_atom)
                index_map[old_idx] = new_idx
            for bond in mol.bonds:
                if bond.atom1 in index_map and bond.atom2 in index_map:
                    if (bond.atom1, bond.atom2) not in cut_pairs:
                        new_bond = Bond(
                            atom1=index_map[bond.atom1],
                            atom2=index_map[bond.atom2],
                            order=bond.order, in_ring=bond.in_ring,
                        )
                        frag.bonds.append(new_bond)
            fragment_mols.append(frag)
        return fragment_mols

    def place_fragments_with_direction(
        self, fragments: List[Molecule], direction: np.ndarray,
        rotation: np.ndarray = None
    ) -> Tuple[Molecule, dict]:
        """Place fragments using a specified direction (Fibonacci sampling).

        Args:
            fragments: List of fragment molecules
            direction: Unit direction vector for placement
            rotation: Optional deterministic rotation matrix. If provided,
                      uses this single rotation instead of random attempts.
        """
        info = {'fragments': [], 'placement': {},
                'direction': direction.tolist()}

        if len(fragments) == 1:
            frag = fragments[0]
            center = self._get_center(frag)
            R = self._get_bounding_radius(frag, center)
            info['fragments'].append({
                'id': 0, 'n_atoms': frag.n_atoms,
                'radius': R, 'center': center.tolist(),
            })
            return fragments[0].copy(), info

        combined = Molecule(charge=0, multiplicity=1)

        # Place first fragment at origin
        frag0 = fragments[0]
        center0 = self._get_center(frag0)
        R_A = self._get_bounding_radius(frag0, center0)
        info['fragments'].append({
            'id': 0, 'n_atoms': frag0.n_atoms,
            'radius': R_A, 'center': [0.0, 0.0, 0.0],
        })

        coords_A = []
        for atom in frag0.atoms:
            new_x = atom.x - center0[0]
            new_y = atom.y - center0[1]
            new_z = atom.z - center0[2]
            new_atom = Atom(atom.element, new_x, new_y, new_z,
                            len(combined.atoms), atom.original_index)
            combined.atoms.append(new_atom)
            coords_A.append((atom.element, np.array([new_x, new_y, new_z])))

        total_attempts = 0
        best_gap = None
        best_dist = None
        all_overlap_resolved = True

        # Place remaining fragments along direction
        for frag_idx, frag in enumerate(fragments[1:], start=1):
            center_B = self._get_center(frag)
            R_B = self._get_bounding_radius(frag, center_B)
            best_placement = None
            best_fallback = None
            min_overlap_count = float('inf')

            # Deterministic rotation: 1 attempt; random: multiple attempts
            n_attempts = 1 if rotation is not None else self.placement_attempts

            for gap_frac in DEFAULT_GAP_FRACTIONS:
                gap = -R_B * gap_frac
                dist = R_A + R_B + gap
                displacement = dist * direction

                for attempt in range(n_attempts):
                    total_attempts += 1
                    rot_matrix = (rotation if rotation is not None
                                  else self._random_rotation())
                    coords_B_placed = []
                    for atom in frag.atoms:
                        pos = np.array([atom.x - center_B[0],
                                        atom.y - center_B[1],
                                        atom.z - center_B[2]])
                        pos = rot_matrix @ pos
                        pos = pos + displacement
                        coords_B_placed.append((atom.element, pos))

                    if not self._has_overlap(coords_A, coords_B_placed):
                        best_placement = (coords_B_placed, rot_matrix,
                                          displacement)
                        best_gap = gap
                        best_dist = dist
                        break
                    else:
                        n_overlaps = self._count_overlaps(
                            coords_A, coords_B_placed
                        )
                        if n_overlaps < min_overlap_count:
                            min_overlap_count = n_overlaps
                            best_fallback = (coords_B_placed, rot_matrix,
                                             displacement)
                            best_gap = gap
                            best_dist = dist

                if best_placement is not None:
                    break

            if best_placement is None:
                best_placement = best_fallback
                all_overlap_resolved = False

            coords_B_placed, rot_matrix, displacement = best_placement
            info['fragments'].append({
                'id': frag_idx, 'n_atoms': frag.n_atoms,
                'radius': R_B, 'center': displacement.tolist(),
            })

            for i, atom in enumerate(frag.atoms):
                elem, pos = coords_B_placed[i]
                new_atom = Atom(elem, pos[0], pos[1], pos[2],
                                len(combined.atoms), atom.original_index)
                combined.atoms.append(new_atom)
                coords_A.append((elem, pos))

            R_A = self._get_bounding_radius_from_coords(coords_A)

        info['placement'] = {
            'gap': best_gap if best_gap else 0.0,
            'dist': best_dist if best_dist else 0.0,
            'attempts': total_attempts,
            'overlap_resolved': all_overlap_resolved,
        }
        return combined, info

    # ---- geometry helpers ----

    def _get_center(self, mol: Molecule) -> np.ndarray:
        coords = np.array([[a.x, a.y, a.z] for a in mol.atoms])
        return coords.mean(axis=0)

    def _get_bounding_radius(self, mol: Molecule,
                              center: np.ndarray) -> float:
        max_dist = 0.0
        for atom in mol.atoms:
            dist = np.linalg.norm(
                np.array([atom.x, atom.y, atom.z]) - center
            )
            max_dist = max(max_dist, dist)
        return max(max_dist, 0.5)

    def _get_bounding_radius_from_coords(
        self, coords_list: List[Tuple[str, np.ndarray]]
    ) -> float:
        if not coords_list:
            return 0.5
        all_coords = np.array([c[1] for c in coords_list])
        center = all_coords.mean(axis=0)
        max_dist = max(
            np.linalg.norm(coord - center) for _, coord in coords_list
        )
        return max(max_dist, 0.5)

    def _has_overlap(self, coords_A, coords_B,
                      scale: float = 0.7) -> bool:
        for elem_a, pos_a in coords_A:
            r_a = COVALENT_RADII.get(elem_a, 1.5)
            for elem_b, pos_b in coords_B:
                r_b = COVALENT_RADII.get(elem_b, 1.5)
                if np.linalg.norm(pos_a - pos_b) < (r_a + r_b) * scale:
                    return True
        return False

    def _count_overlaps(self, coords_A, coords_B,
                         scale: float = 0.7) -> int:
        count = 0
        for elem_a, pos_a in coords_A:
            r_a = COVALENT_RADII.get(elem_a, 1.5)
            for elem_b, pos_b in coords_B:
                r_b = COVALENT_RADII.get(elem_b, 1.5)
                if np.linalg.norm(pos_a - pos_b) < (r_a + r_b) * scale:
                    count += 1
        return count

    def _random_rotation(self) -> np.ndarray:
        u1, u2, u3 = self.rng.random(3)
        q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        R = np.array([
            [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3),
             2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3),
             2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1),
             1 - 2*(q1*q1 + q2*q2)],
        ])
        return R

    @staticmethod
    def _rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotation matrix around an arbitrary axis by angle (radians).

        Uses Rodrigues' rotation formula:
            R = cos(a)I + sin(a)[u]x + (1-cos(a))(u outer u)
        """
        u = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c
        x, y, z = u
        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
        ])

# ============================================================
# Stability Checker
# ============================================================
class StabilityChecker:
    """Check structural validity after optimization."""

    def __init__(self, params: SFRISParams,
                 reference_energy: Optional[float] = None):
        self.params = params
        self.reference_energy = reference_energy

    def check_connectivity(self, mol: Molecule) -> bool:
        if mol.n_atoms == 0:
            return False
        adjacency = {i: [] for i in range(mol.n_atoms)}
        for i, atom1 in enumerate(mol.atoms):
            for j, atom2 in enumerate(mol.atoms):
                if j <= i:
                    continue
                dist = atom1.distance_to(atom2)
                r1 = COVALENT_RADII.get(atom1.element, 1.5)
                r2 = COVALENT_RADII.get(atom2.element, 1.5)
                if dist < (r1 + r2) * BOND_THRESHOLD:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        visited = set([0])
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == mol.n_atoms

    def check_atom_count(self, mol: Molecule, reference: Molecule) -> bool:
        if mol.n_atoms != reference.n_atoms:
            return False

        def get_counts(m):
            counts = {}
            for atom in m.atoms:
                counts[atom.element] = counts.get(atom.element, 0) + 1
            return counts

        return get_counts(mol) == get_counts(reference)

    def check_energy(self, energy: float) -> bool:
        if self.reference_energy is None or energy is None:
            return True
        delta_e = (energy - self.reference_energy) * 627.509
        return delta_e < self.params.energy_cutoff

    def is_stable(self, mol: Molecule, reference: Molecule,
                  energy: float = None) -> Tuple[bool, str]:
        if not self.check_connectivity(mol):
            return False, "DISSOCIATED"
        if not self.check_atom_count(mol, reference):
            return False, "ATOM_COUNT_MISMATCH"
        if energy is not None and not self.check_energy(energy):
            return False, "ENERGY_TOO_HIGH"
        return True, "OK"

# ============================================================
# Structure Clusterer
# ============================================================
class StructureClusterer:
    """Cluster structures using hierarchical complete-linkage clustering."""

    def __init__(self, rmsd_threshold: float = 0.5,
                 energy_threshold: float = 1.0, heavy_only: bool = True,
                 mirror_check: bool = False):
        self.rmsd_threshold = rmsd_threshold
        self.energy_threshold = energy_threshold
        self.energy_threshold_eh = energy_threshold / 627.509
        self.heavy_only = heavy_only
        self.mirror_check = mirror_check

    def cluster(self, molecules: List[Molecule],
                logger=None) -> Tuple[List[List[int]], np.ndarray]:
        n = len(molecules)
        if n == 0:
            return [], np.array([])
        if n == 1:
            return [[0]], np.array([[0.0]])

        if logger:
            heavy_str = "heavy-atom" if self.heavy_only else "all-atom"
            logger.info(
                f"Clustering {n} structures ({heavy_str} RMSD "
                f"< {self.rmsd_threshold} A, "
                f"dE < {self.energy_threshold} kcal/mol)"
            )

        # Build pairwise distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_distance(molecules[i], molecules[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Complete-linkage hierarchical clustering
        condensed = squareform(dist_matrix)
        max_finite = (condensed[np.isfinite(condensed)].max()
                      if np.any(np.isfinite(condensed)) else 1.0)
        large_dist = max_finite * 10 + 100
        condensed_for_linkage = np.where(
            np.isinf(condensed), large_dist, condensed
        )

        Z = linkage(condensed_for_linkage, method='complete')
        labels = fcluster(Z, t=self.rmsd_threshold, criterion='distance')

        clusters_dict = {}
        for idx, label in enumerate(labels):
            clusters_dict.setdefault(label, []).append(idx)
        clusters = list(clusters_dict.values())

        def cluster_min_energy(cluster_indices):
            energies = [molecules[i].energy for i in cluster_indices
                        if molecules[i].energy is not None]
            return min(energies) if energies else float('inf')

        clusters.sort(key=cluster_min_energy)

        if logger:
            logger.info(
                f"Clustering complete: {n} structures -> {len(clusters)} clusters"
            )
        return clusters, dist_matrix

    def _compute_distance(self, mol1: Molecule, mol2: Molecule) -> float:
        if mol1.energy is not None and mol2.energy is not None:
            if abs(mol1.energy - mol2.energy) > self.energy_threshold_eh:
                return float('inf')
        return self._calculate_rmsd_kabsch(mol1, mol2,
                                            heavy_only=self.heavy_only)

    def _calculate_rmsd_kabsch(self, mol1: Molecule, mol2: Molecule,
                                heavy_only: bool = False) -> float:
        if heavy_only:
            atoms1 = [(i, a) for i, a in enumerate(mol1.atoms)
                      if a.element != 'H']
            atoms2 = [(i, a) for i, a in enumerate(mol2.atoms)
                      if a.element != 'H']
        else:
            atoms1 = list(enumerate(mol1.atoms))
            atoms2 = list(enumerate(mol2.atoms))

        n_atoms = len(atoms1)
        if n_atoms == 0:
            return 0.0
        if n_atoms != len(atoms2):
            return float('inf')

        # Group by element
        elements1 = {}
        elements2 = {}
        for idx, a in atoms1:
            elements1.setdefault(a.element, []).append((idx, a))
        for idx, a in atoms2:
            elements2.setdefault(a.element, []).append((idx, a))
        if set(elements1.keys()) != set(elements2.keys()):
            return float('inf')
        for elem in elements1:
            if len(elements1[elem]) != len(elements2.get(elem, [])):
                return float('inf')

        # Center coordinates
        coords1 = np.array([[a.x, a.y, a.z] for _, a in atoms1])
        coords2 = np.array([[a.x, a.y, a.z] for _, a in atoms2])
        coords1 = coords1 - coords1.mean(axis=0)
        coords2 = coords2 - coords2.mean(axis=0)

        # Local index maps
        local_idx1 = {orig_idx: local
                      for local, (orig_idx, _) in enumerate(atoms1)}
        local_idx2 = {orig_idx: local
                      for local, (orig_idx, _) in enumerate(atoms2)}

        def _hungarian_kabsch_rmsd(c1, c2):
            """Full Hungarian mapping + Kabsch alignment RMSD."""
            mapping = {}
            for elem in elements1:
                elem_atoms1 = elements1[elem]
                elem_atoms2 = elements2[elem]
                if len(elem_atoms1) == 1:
                    mapping[local_idx1[elem_atoms1[0][0]]] = \
                        local_idx2[elem_atoms2[0][0]]
                else:
                    cost = np.zeros((len(elem_atoms1), len(elem_atoms2)))
                    for i, (idx1, _) in enumerate(elem_atoms1):
                        for j, (idx2, _) in enumerate(elem_atoms2):
                            l1 = local_idx1[idx1]
                            l2 = local_idx2[idx2]
                            cost[i, j] = np.linalg.norm(c1[l1] - c2[l2])
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c_idx in zip(row_ind, col_ind):
                        l1 = local_idx1[elem_atoms1[r][0]]
                        l2 = local_idx2[elem_atoms2[c_idx][0]]
                        mapping[l1] = l2
            # Reorder c1 to match c2 ordering
            c1_matched = np.zeros_like(c1)
            for l1, l2 in mapping.items():
                c1_matched[l2] = c1[l1]
            # Kabsch rotation
            H = c1_matched.T @ c2
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            c1_aligned = c1_matched @ R
            diff = c1_aligned - c2
            return np.sqrt((diff ** 2).sum() / n_atoms)

        rmsd_normal = _hungarian_kabsch_rmsd(coords1, coords2)

        # Mirror check: reflect x-axis, redo full mapping + alignment
        if self.mirror_check:
            coords1_mirror = coords1.copy()
            coords1_mirror[:, 0] *= -1
            rmsd_mirror = _hungarian_kabsch_rmsd(coords1_mirror, coords2)
            return min(rmsd_normal, rmsd_mirror)

        return rmsd_normal

    def select_representatives(
        self, molecules: List[Molecule], clusters: List[List[int]]
    ) -> List[Tuple[int, Molecule]]:
        representatives = []
        for cluster in clusters:
            best_idx = None
            best_energy = float('inf')
            for idx in cluster:
                mol = molecules[idx]
                if mol.energy is not None and mol.energy < best_energy:
                    best_energy = mol.energy
                    best_idx = idx
            if best_idx is None:
                best_idx = cluster[0]
            representatives.append((best_idx, molecules[best_idx]))
        representatives.sort(
            key=lambda x: x[1].energy
            if x[1].energy is not None else float('inf')
        )
        return representatives