"""TDDFT, spin-orbit and ESD results in ORCA 6 outputs, for the ESD category."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

EH_TO_CM = 219474.63
EV_TO_EH = 1 / 27.211386

_FINAL = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_SOC_CIS = re.compile(r"E\(SOC CIS\)\s*=\s*(-?\d+\.\d+) Eh")
_DE_CIS = re.compile(r"DE\(CIS\)\s*=\s*(-?\d+\.\d+) Eh")
_DE_CIS_ROOT = re.compile(r"DE\(CIS\)\s*=\s*-?\d+\.\d+ Eh \(Root\s*(\d+)\)")
_ROOTS_TITLE = re.compile(r"^TD-DFT(?:/TDA)? EXCITED STATES \((SINGLETS|TRIPLETS)\)", re.MULTILINE)
_ROOT = re.compile(r"^STATE\s+\d+:\s+E=\s+(-?\d+\.\d+) au", re.MULTILINE)
_SOCME_TITLE = "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS"
_SOCME_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\(.*\))\s*$")
_NUMBER = re.compile(r"-?\d+\.\d+")
_ESD_DONE = "ORCA ESD FINISHED WITHOUT ERROR"
_RATE = re.compile(r"^The calculated (.+?) rate constant is\s+(\S+) s-1", re.MULTILINE)
# HT splits print "25.67 from FC" for ISC but "100.00% from FC" for emission.
_SPLIT = re.compile(r"^with\s+(-?[\d.]+)%? from FC and\s+(-?[\d.]+)%? from HT", re.MULTILINE)
_K_SQUARED = re.compile(r"^The sum of K\*K is:\s+(\S+)", re.MULTILINE)
_E00 = re.compile(r"^0-0 energy difference:\s+(-?[\d.]+) cm-1", re.MULTILINE)
_LIBXC = re.compile(r"Third functional derivative of a B88|Native implementation of 3d derivative")


@dataclass(frozen=True)
class Rate:
    """One ESD rate constant as ORCA printed it."""

    process: str
    rate: float
    fc_percent: Optional[float] = None
    ht_percent: Optional[float] = None
    k_squared: Optional[float] = None
    e00_cm: Optional[float] = None


def ground_energy(content: str) -> Optional[float]:
    """Ground-state energy (Eh) of a TDDFT singlepoint.

    FINAL SINGLE POINT ENERGY includes the followed root (with DOSOC, SOC root 1).
    """
    finals = _FINAL.findall(content)
    if not finals:
        return None
    shift = _SOC_CIS.findall(content) or _DE_CIS.findall(content)
    return float(finals[-1]) - (float(shift[-1]) if shift else 0.0)


def excitation_energies(content: str, kind: str) -> dict[int, float]:
    """``{n: excitation energy (Eh)}`` of the last SINGLETS or TRIPLETS table.

    Numbered in table order; TDA numbers its triplets after the singlets.
    """
    titles = list(_ROOTS_TITLE.finditer(content))
    wanted = [i for i, title in enumerate(titles) if title.group(1) == kind]
    if not wanted:
        return {}
    i = wanted[-1]
    end = titles[i + 1].start() if i + 1 < len(titles) else len(content)
    energies = _ROOT.findall(content, titles[i].end(), end)
    return {n: float(e) for n, e in enumerate(energies, 1)}


def followed_root_energy(content: str) -> Optional[float]:
    """Excitation energy (Eh) of the root a TDDFT job followed, from the last ``DE(CIS)``."""
    found = _DE_CIS.findall(content)
    return float(found[-1]) if found else None


def followed_root(content: str) -> Optional[int]:
    """The N of the last ``DE(CIS) = ... Eh (Root N)`` line."""
    found = _DE_CIS_ROOT.findall(content)
    return int(found[-1]) if found else None


def socme(content: str) -> dict[tuple[int, int], float]:
    """``{(T, S): |<T|H_SO|S>|}`` in cm-1 from the last SOCME table; S = 0 is the ground state."""
    if _SOCME_TITLE not in content:
        return {}
    table: dict[tuple[int, int], float] = {}
    for line in content.rsplit(_SOCME_TITLE, 1)[1].splitlines():
        row = _SOCME_ROW.match(line)
        if row is None:
            if table:
                break
            continue
        values = [float(v) for v in _NUMBER.findall(row.group(3))]
        table[(int(row.group(1)), int(row.group(2)))] = math.sqrt(sum(v * v for v in values))
    return table


def rates(content: str) -> list[Rate]:
    """Every ESD rate, one per ORCA job (``$new_job``), in order."""
    found: list[Rate] = []
    for job in content.split(_ESD_DONE)[:-1]:
        rate = _RATE.findall(job)
        if not rate:
            continue
        split = _SPLIT.findall(job)
        k_squared = _K_SQUARED.findall(job)
        e00 = _E00.findall(job)
        found.append(Rate(
            process=rate[-1][0],
            rate=float(rate[-1][1]),
            fc_percent=float(split[-1][0]) if split else None,
            ht_percent=float(split[-1][1]) if split else None,
            k_squared=float(k_squared[-1]) if k_squared else None,
            e00_cm=float(e00[-1]) if e00 else None,
        ))
    return found


def needs_libxc(content: str) -> bool:
    """Whether ORCA refused a native B88 functional's third derivatives."""
    return _LIBXC.search(content) is not None
