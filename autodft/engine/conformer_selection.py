"""Pick a structurally diverse subset of a conformer ensemble.

The minimum is always kept; the rest are chosen by farthest-point sampling
on symmetry-aware heavy-atom RMSD, among conformers within *window* kcal/mol
of the minimum.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# The metadata key holding the window in kcal/mol; absent means keep the N lowest.
DIVERSITY_WINDOW = "conformer_diversity_window"


def diverse(xyzs: list[str], energies: list[float], n: int, window: float) -> Optional[list[int]]:
    """Indices of the minimum plus the n-1 most diverse conformers within *window*, in energy order.

    None when the geometries cannot be compared.
    """
    if n < 1 or not xyzs:
        return []
    order = sorted(range(len(xyzs)), key=lambda i: energies[i])
    lowest = energies[order[0]]
    candidates = [i for i in order if energies[i] - lowest <= window]
    if len(candidates) <= n:
        return candidates

    mol = _heavy_atom_conformers([xyzs[i] for i in candidates])
    if mol is None:
        return None

    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    ref, probe = Chem.Mol(mol), Chem.Mol(mol)
    picked = [0]
    nearest = [rdMolAlign.GetBestRMS(probe, ref, c, 0) for c in range(len(candidates))]
    while len(picked) < n:
        # Ties go to the lower energy (candidates are in energy order).
        best = max((c for c in range(len(candidates)) if c not in picked), key=lambda c: (nearest[c], -c))
        picked.append(best)
        for c in range(len(candidates)):
            if c not in picked:
                nearest[c] = min(nearest[c], rdMolAlign.GetBestRMS(probe, ref, c, best))
    return [candidates[c] for c in sorted(picked)]


def _heavy_atom_conformers(xyzs: list[str]):
    """One heavy-atom mol with every xyz as a conformer; None if they don't share one atom list."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDetermineBonds

    quiet = rdBase.BlockLogs()  # noqa: F841 - logs stay blocked while it lives
    try:
        mol = Chem.MolFromXYZBlock(xyzs[0])
        rdDetermineBonds.DetermineConnectivity(mol)
        elements = [a.GetAtomicNum() for a in mol.GetAtoms()]
        mol.RemoveAllConformers()
        for xyz in xyzs:
            other = Chem.MolFromXYZBlock(xyz)
            if other is None or [a.GetAtomicNum() for a in other.GetAtoms()] != elements:
                return None
            mol.AddConformer(Chem.Conformer(other.GetConformer()), assignId=True)
        heavy = Chem.RemoveHs(mol, sanitize=False)
        Chem.FastFindRings(heavy)
        return heavy
    except Exception:  # noqa: BLE001 - an unreadable ensemble falls back to the N lowest
        logger.exception("Could not compare conformers")
        return None
