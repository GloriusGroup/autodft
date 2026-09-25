"""Spectra from ORCA output: TDDFT absorption and IR tables.

Only the plain electric-dipole absorption table is read. With DOSOC, ORCA also
prints SOC-corrected and velocity-gauge tables in the same row format; the
parser keys on the exact title and stops at the first table's end. With
``triplets true`` the plain table also lists spin-forbidden rows
(``0-1A -> 1-3A``, fosc 0); only spin-allowed rows are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

EV_TO_CM = 8065.543937

_ABS_TITLE = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
# ORCA 6: "  0-1A  ->  3-1A    6.022348   48573.5   205.9   0.240844161 ..."
# Groups: initial multiplicity, final root, final multiplicity, eV, cm-1, nm, fosc.
_ABS_ROW_6 = re.compile(
    r"^\s*\d+-(\d+)\w*\s+->\s+(\d+)-(\d+)\w*\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)"
)
# ORCA 5: "   3   48573.5    205.9   0.240844161 ..."
_ABS_ROW_5 = re.compile(r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([-\d.eE+]+)\s")
# "  6:    245.98   0.000004    0.02  0.000005  ( ... )"
_IR_ROW = re.compile(r"^\s*(\d+):\s+(-?[\d.]+)\s+[-\d.eE+]+\s+([-\d.eE+]+)\s")


@dataclass(frozen=True)
class Transition:
    root: int
    energy_ev: float
    energy_cm: float
    wavelength_nm: float
    fosc: float


@dataclass(frozen=True)
class IRMode:
    mode: int
    frequency_cm: float
    intensity_km_mol: float


def parse_absorption(content: str) -> list[Transition]:
    """Transitions from the first plain electric-dipole absorption table."""
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _ABS_TITLE), None)
    if start is None:
        return []

    rows: list[Transition] = []
    seen_row = False
    for line in lines[start + 1:]:
        match6 = _ABS_ROW_6.match(line)
        if match6:
            seen_row = True
            mult_from, root, mult_to, ev, cm, nm, fosc = match6.groups()
            if mult_to == mult_from:
                rows.append(Transition(int(root), float(ev), float(cm), float(nm), float(fosc)))
            continue
        match5 = _ABS_ROW_5.match(line)
        if match5:
            seen_row = True
            root, cm, nm, fosc = match5.groups()
            rows.append(Transition(int(root), float(cm) / EV_TO_CM, float(cm), float(nm), float(fosc)))
            continue
        if seen_row and not line.strip():
            break
    return rows


def parse_ir(content: str) -> list[IRMode]:
    """Modes from the last IR SPECTRUM table."""
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "IR SPECTRUM"]
    if not starts:
        return []

    modes: list[IRMode] = []
    for line in lines[starts[-1] + 1:]:
        match = _IR_ROW.match(line)
        if match:
            modes.append(IRMode(int(match.group(1)), float(match.group(2)), float(match.group(3))))
            continue
        if modes and not line.strip():
            break
    return modes


_SHIELDING_TITLE = "CHEMICAL SHIELDING SUMMARY (ppm)"
# "      2       C           67.627        182.501 "
_SHIELDING_ROW = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")


@dataclass(frozen=True)
class Shielding:
    index: int  # 0-based ORCA atom index
    element: str
    isotropic: float
    anisotropy: float


def parse_shieldings(content: str) -> list[Shielding]:
    """Isotropic shieldings (ppm) from the last shielding summary.

    Double hybrids print three -- SCF, unrelaxed and relaxed MP2 -- and the
    last is the final result.
    """
    lines = content.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == _SHIELDING_TITLE]
    if not starts:
        return []

    rows: list[Shielding] = []
    for line in lines[starts[-1] + 1:]:
        match = _SHIELDING_ROW.match(line)
        if match:
            rows.append(Shielding(
                int(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4)),
            ))
            continue
        if rows:
            break
    return rows


_NPA_TITLE = "Summary of Natural Population Analysis:"
# "    C  1    0.30354      1.99995     3.66683    0.02968     5.69646" -- the
# open-shell total block adds a Natural Spin Density column at the end.
# NBO 7's fixed-width format (1x,2x,a2,i3,...) runs element and number
# together from atom 100 on, e.g. "    C100    0.30354 ...".
_NPA_ROW = re.compile(
    r"^\s*([A-Z][a-z]?)\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)"
    r"(?:\s+(-?[\d.]+))?\s*$"
)
_ATOM_COUNT_RE = re.compile(r"Number of atoms\s*\.+\s*(\d+)")


@dataclass(frozen=True)
class NaturalCharge:
    index: int  # 0-based: the NBO atom number - 1
    element: str
    charge: float
    spin: Optional[float] = None  # Natural Spin Density; only for open shell


def parse_natural_charges(content: str) -> list[NaturalCharge]:
    """Atoms from the first ``Summary of Natural Population Analysis`` block.

    Stops at that block's ``====`` line, so the total row and any later
    (alpha/beta) summaries are never read. When ORCA's own atom count is
    printed and disagrees with what was parsed, ``[]`` is returned instead
    of a silently truncated list.
    """
    lines = content.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == _NPA_TITLE), None)
    if start is None:
        return []

    rows: list[NaturalCharge] = []
    for line in lines[start + 1:]:
        if line.strip().startswith("===="):
            break
        match = _NPA_ROW.match(line)
        if match:
            element, atom_no, charge, _core, _valence, _rydberg, _total, spin = match.groups()
            rows.append(NaturalCharge(
                int(atom_no) - 1, element, float(charge), float(spin) if spin is not None else None,
            ))
    count_match = _ATOM_COUNT_RE.search(content)
    if count_match is not None and len(rows) != int(count_match.group(1)):
        return []
    return rows
