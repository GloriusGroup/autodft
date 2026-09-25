"""Whether an optimised geometry still has its molecule's bonds.

Bonds are perceived from covalent radii (RDKit's connect-the-dots) and
compared, as a graph of elements and bonds, with the SMILES topology --
bond orders, charges and atom order play no part.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


def check(smiles: str, xyz: str) -> Optional[dict]:
    """``{"ok", "fragments", "detail"}`` for *xyz* against *smiles*; None when they cannot be compared."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDetermineBonds

    from autodft.engine.entrypoint_processor import mol_from_smiles

    quiet = rdBase.BlockLogs()  # noqa: F841 - logs stay blocked while it lives
    try:
        reference = mol_from_smiles(smiles)
        geometry = Chem.MolFromXYZBlock(xyz)
        if reference is None or geometry is None:
            return None
        reference = _graph(Chem.AddHs(reference))
        rdDetermineBonds.DetermineConnectivity(geometry)
        geometry = _graph(geometry)
    except Exception:  # noqa: BLE001 - an unreadable input is unchecked, not flagged
        return None
    if _formula(reference) != _formula(geometry):
        return None

    fragments = len(Chem.GetMolFrags(geometry))
    ok = (
        reference.GetNumBonds() == geometry.GetNumBonds()
        and geometry.HasSubstructMatch(reference)
    )
    if ok:
        return {"ok": True, "fragments": fragments, "detail": ""}

    changes = _changed_bonds(reference, geometry) or "bonds rearranged"
    if fragments > len(Chem.GetMolFrags(reference)):
        pieces = " + ".join(
            _formula(geometry, atoms) for atoms in sorted(Chem.GetMolFrags(geometry), key=len, reverse=True)
        )
        detail = f"dissociated: {fragments} fragments ({pieces}); {changes}"
    else:
        detail = f"bonds changed: {changes}"
    return {"ok": False, "fragments": fragments, "detail": detail}


def _graph(mol):
    """*mol* reduced to elements and single bonds."""
    from rdkit import Chem

    graph = Chem.RWMol()
    for atom in mol.GetAtoms():
        plain = Chem.Atom(atom.GetAtomicNum())
        plain.SetNoImplicit(True)
        graph.AddAtom(plain)
    for bond in mol.GetBonds():
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    graph = graph.GetMol()
    graph.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(graph)
    return graph


def _formula(mol, atoms=None) -> str:
    """Hill formula of *mol*, or of the atoms *atoms* of it."""
    symbols = Counter(
        mol.GetAtomWithIdx(i).GetSymbol() for i in (atoms if atoms is not None else range(mol.GetNumAtoms()))
    )
    first = [s for s in ("C", "H") if "C" in symbols and s in symbols]
    order = first + sorted(s for s in symbols if s not in first)
    return "".join(f"{s}{symbols[s] if symbols[s] > 1 else ''}" for s in order)


def _changed_bonds(reference, geometry) -> str:
    """Bond counts per element pair that differ, e.g. ``C-H bonds 9 -> 8``."""
    def counts(mol):
        return Counter(
            "-".join(sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()))) for b in mol.GetBonds()
        )

    before, after = counts(reference), counts(geometry)
    return ", ".join(
        f"{pair} bonds {before[pair]} -> {after[pair]}"
        for pair in sorted(set(before) | set(after)) if before[pair] != after[pair]
    )
