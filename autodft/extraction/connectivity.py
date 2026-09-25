"""Whether an optimised geometry still has its molecule's bonds.

Bonds are perceived from covalent radii (RDKit's connect-the-dots) and
compared, as a graph of elements and bonds, with the SMILES topology --
bond orders, charges, atom order and bonds to metals play no part.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


# Bonds to these are ionic or dative and drawn inconsistently in SMILES
# (Na+ acetate, ionic ferrocene), so neither graph keeps them.
METALS = frozenset(
    [3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 87, 88]
    + list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81)) + list(range(89, 113))
)


def check(smiles: str, xyz: str) -> Optional[dict]:
    """``{"ok", "lost", "fragments", "detail"}`` for *xyz* against *smiles*; None when they cannot be compared.

    ``lost`` is True when a SMILES bond is missing (dissociation, H shift); a
    geometry that only formed extra bonds is not ok but has ``lost`` False.
    """
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
    before, after = _bond_counts(reference), _bond_counts(geometry)
    # A pair with fewer bonds lost one for sure; otherwise every SMILES bond
    # must still map onto the geometry.
    lost = any(after[pair] < n for pair, n in before.items()) or not geometry.HasSubstructMatch(reference)
    changes = ", ".join(
        f"{pair} bonds {before[pair]} -> {after[pair]}"
        for pair in sorted(set(before) | set(after)) if before[pair] != after[pair]
    )
    if not lost and not changes:
        return {"ok": True, "lost": False, "fragments": fragments, "detail": ""}

    if not lost:
        detail = f"bond formed: {changes}"
    elif fragments > len(Chem.GetMolFrags(reference)):
        pieces = " + ".join(
            _formula(geometry, atoms) for atoms in sorted(Chem.GetMolFrags(geometry), key=len, reverse=True)
        )
        detail = f"dissociated: {fragments} fragments ({pieces}); {changes or 'bonds rearranged'}"
    else:
        detail = f"bonds changed: {changes or 'bonds rearranged'}"
    return {"ok": False, "lost": lost, "fragments": fragments, "detail": detail}


def breaks(found: Optional[dict]) -> bool:
    """Whether a stored check lost bonds of the SMILES -- what ESD refuses to build on."""
    return found is not None and bool(found.get("lost"))


def _graph(mol):
    """*mol* reduced to elements and single bonds, without bonds to metals."""
    from rdkit import Chem

    graph = Chem.RWMol()
    for atom in mol.GetAtoms():
        plain = Chem.Atom(atom.GetAtomicNum())
        plain.SetNoImplicit(True)
        graph.AddAtom(plain)
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() in METALS or bond.GetEndAtom().GetAtomicNum() in METALS:
            continue
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


def _bond_counts(mol) -> Counter:
    """Bonds per element pair, e.g. ``{"C-H": 9}``."""
    return Counter(
        "-".join(sorted((b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()))) for b in mol.GetBonds()
    )
