"""Per-molecule state analysis.

Computes, per molecule and per conformer-selection mode:

  - Triplet energy             ΔE(T1) = E_comb(T1) − E_comb(S0)
  - T1 reorganisation energy   full 4-point Marcus on the S0↔T1 surface,
                               requires e_vert_spin_change on BOTH sides
  - Redox free energies        ΔG_ox = G(ox) − G(S0), ΔG_red = G(S0) − G(red)
  - E vs SCE in MeCN           ΔG/F − E_ref_SCE  (only when MeCN solvation
                               is detected in any header)
  - Redox reorganisation       4-point Marcus on each S0↔ox / S0↔red surface,
                               requires the appropriate vertical SP on
                               both ends

Two conformer-selection modes:

  - ``lowest_energy``: per state, pick the conformer with the smallest
    free-energy-corrected E_combined.
  - ``rmsd_matched``: S0 keeps its lowest-energy conformer; every other
    state picks the conformer with the smallest Kabsch RMSD relative to
    the S0 reference geometry. Direct atom-by-atom RMSD is appropriate
    because every state starts from the same conformer seeds, so the
    optimised geometries preserve atom ordering.

Reference values:

  - ``SCE_ABS_MECN = 4.42 V`` (Pavlishchuk & Addison, *Inorg. Chim. Acta*
    2000, 298, 97-102).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlmodel import col, select

from autodft.db import get_session
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import (
    ComputationHeader,
    ComputationTask,
    Molecule,
    MoleculeState,
)
from autodft.models.geometry import MoleculeGeometry


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_PER_MOL = 627.5094740631
SCE_ABS_MECN = 4.42  # V, absolute SCE potential in MeCN (Pavlishchuk & Addison 2000)


# Match MeCN solvation written in any of the common ORCA forms:
#   "! CPCM(Acetonitrile)" / "! SMD(MeCN)" / "%cpcm  smd true  SMDsolvent \"acetonitrile\" end"
_MECN_RE = re.compile(
    r"(CPCM|SMD)\s*\(\s*(?:acetonitrile|mecn)\s*\)"
    r"|smdsolvent\s*[=\s]*[\"']?\s*(?:acetonitrile|mecn)"
    r"|solvent\s*[=\s]+[\"']?\s*(?:acetonitrile|mecn)",
    re.IGNORECASE,
)


def detects_mecn_solvation(header_text: Optional[str]) -> bool:
    return bool(header_text and _MECN_RE.search(header_text))


# ----------------------------------------------------------------------
# Geometry parsing + Kabsch RMSD
# ----------------------------------------------------------------------


def parse_xyz(xyz_data: str) -> tuple[list[str], np.ndarray]:
    """Parse an XYZ string into element symbols + (N, 3) coords.

    Accepts both the standard ``.xyz`` layout (count + comment + rows)
    and a bare ``symbol x y z`` table.
    """
    if not xyz_data:
        return [], np.empty((0, 3))
    lines = xyz_data.strip().splitlines()
    head = lines[0].strip().split() if lines else []
    if len(head) == 1 and head[0].isdigit():
        # Standard .xyz: count line + comment line + atom rows.
        lines = lines[2:] if len(lines) >= 2 else []
    elems: list[str] = []
    coords: list[list[float]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        elems.append(parts[0])
    if not coords:
        return elems, np.empty((0, 3))
    return elems, np.asarray(coords, dtype=float)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """Kabsch-aligned RMSD in Å between two (N, 3) coordinate arrays.

    Caller must ensure both arrays share atom ordering — no atom mapping
    is performed.
    """
    if p.shape != q.shape or p.size == 0:
        raise ValueError("coordinate arrays must share a non-empty shape")
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = p_c.T @ q_c
    u, _, vt = np.linalg.svd(h)
    d = float(np.sign(np.linalg.det(vt.T @ u.T)))
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T
    rotated = p_c @ rot.T
    diff = rotated - q_c
    return float(np.sqrt((diff * diff).sum() / p.shape[0]))


# ----------------------------------------------------------------------
# Per-mode analysis
# ----------------------------------------------------------------------


@dataclass
class StatePick:
    state: str
    conformer_index: int
    opt_task_id: int
    rmsd_to_s0: Optional[float]  # Å, None for S0 and when RMSD can't be computed
    e_sp: Optional[float]                # Hartree
    e_correction: Optional[float]        # Hartree
    e_combined: Optional[float]          # Hartree
    e_vert_ox: Optional[float]           # Hartree
    e_vert_red: Optional[float]          # Hartree
    e_vert_spin_change: Optional[float]  # Hartree


@dataclass
class ModeAnalysis:
    mode: str
    picks: dict[str, StatePick]
    triplet_energy_ev: Optional[float] = None
    lambda_t1_ev: Optional[float] = None
    dG_ox_ev: Optional[float] = None
    dG_red_ev: Optional[float] = None
    E_ox_vs_sce: Optional[float] = None
    E_red_vs_sce: Optional[float] = None
    lambda_ox_ev: Optional[float] = None
    lambda_red_ev: Optional[float] = None


def _compute_derived(picks: dict[str, StatePick], mecn: bool) -> ModeAnalysis:
    ana = ModeAnalysis(mode="", picks=picks)
    s0 = picks.get("S0")
    t1 = picks.get("T1")
    ox = picks.get("ox")
    red = picks.get("red")

    if s0 and t1 and s0.e_combined is not None and t1.e_combined is not None:
        ana.triplet_energy_ev = (t1.e_combined - s0.e_combined) * HARTREE_TO_EV

    # Full 4-point Marcus on S0 ↔ T1:
    #   λ = ½ · [(E_vert_T1@S0 − E_sp(T1)) + (E_vert_S0@T1 − E_sp(S0))]
    if (
        s0 and t1
        and s0.e_vert_spin_change is not None and t1.e_sp is not None
        and t1.e_vert_spin_change is not None and s0.e_sp is not None
    ):
        l_a = s0.e_vert_spin_change - t1.e_sp
        l_b = t1.e_vert_spin_change - s0.e_sp
        ana.lambda_t1_ev = 0.5 * (l_a + l_b) * HARTREE_TO_EV

    if s0 and ox and s0.e_combined is not None and ox.e_combined is not None:
        ana.dG_ox_ev = (ox.e_combined - s0.e_combined) * HARTREE_TO_EV
        if mecn:
            ana.E_ox_vs_sce = ana.dG_ox_ev - SCE_ABS_MECN
    if s0 and red and s0.e_combined is not None and red.e_combined is not None:
        ana.dG_red_ev = (s0.e_combined - red.e_combined) * HARTREE_TO_EV
        if mecn:
            ana.E_red_vs_sce = ana.dG_red_ev - SCE_ABS_MECN

    # Redox reorganisation — strict 4-point Marcus. Requires the
    # cross-vertical SP at both endpoints (vert_ox at S0, vert_red at ox;
    # and the reverse for the reduction branch).
    if (
        s0 and ox
        and s0.e_vert_ox is not None and s0.e_sp is not None
        and ox.e_vert_red is not None and ox.e_sp is not None
    ):
        l_s0 = ox.e_vert_red - s0.e_sp
        l_ox = s0.e_vert_ox - ox.e_sp
        ana.lambda_ox_ev = 0.5 * (l_s0 + l_ox) * HARTREE_TO_EV
    if (
        s0 and red
        and s0.e_vert_red is not None and s0.e_sp is not None
        and red.e_vert_ox is not None and red.e_sp is not None
    ):
        l_s0 = red.e_vert_ox - s0.e_sp
        l_red = s0.e_vert_red - red.e_sp
        ana.lambda_red_ev = 0.5 * (l_s0 + l_red) * HARTREE_TO_EV

    return ana


# ----------------------------------------------------------------------
# Project-level driver
# ----------------------------------------------------------------------


def analyze_project(project_name: str) -> dict:
    """Run state analysis for every molecule in a project.

    Heavy lifting (ORCA output parsing) is delegated to
    :class:`PipelineExtractor` so this stays consistent with the CSV/JSON
    exports.
    """
    extractor = PipelineExtractor(project_name)
    all_results: list[ConformerResult] = extractor.extract_results(all_conformers=True)

    # (mol_id, state_desc) -> [ConformerResult, ...], ordered by opt_task_id
    groups: dict[tuple[int, str], list[ConformerResult]] = {}
    for r in all_results:
        groups.setdefault((r.molecule_id, r.state), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.opt_task_id)

    with get_session() as session:
        mols = session.exec(
            select(Molecule)
            .where(Molecule.project_name == project_name)
            .order_by(col(Molecule.id).asc())
        ).all()
        mol_ids = [m.id for m in mols if m.id is not None]
        if not mol_ids:
            return {
                "project": project_name,
                "solvation_mecn": False,
                "sce_ref": SCE_ABS_MECN,
                "molecules": [],
            }

        # MeCN detection: scan every opt / SP header touched by this project.
        states = session.exec(
            select(MoleculeState).where(col(MoleculeState.molecule_id).in_(mol_ids))
        ).all()
        header_ids = {
            hid for st in states
            for hid in (st.optimization_header_id, st.singlepoint_header_id)
            if hid is not None
        }
        headers = (
            session.exec(
                select(ComputationHeader).where(col(ComputationHeader.id).in_(list(header_ids)))
            ).all()
            if header_ids else []
        )
        solvation_mecn = any(detects_mecn_solvation(h.header_text) for h in headers)

        opt_task_ids = [r.opt_task_id for r in all_results]
        opt_tasks = (
            session.exec(
                select(ComputationTask).where(col(ComputationTask.id).in_(opt_task_ids))
            ).all()
            if opt_task_ids else []
        )
        opt_by_id = {t.id: t for t in opt_tasks}
        out_geom_ids = [t.output_geometry_id for t in opt_tasks if t.output_geometry_id]
        out_geoms = (
            session.exec(
                select(MoleculeGeometry).where(col(MoleculeGeometry.id).in_(out_geom_ids))
            ).all()
            if out_geom_ids else []
        )
        geom_by_id = {g.id: g for g in out_geoms}

    def _coords_for(opt_task_id: int) -> Optional[np.ndarray]:
        task = opt_by_id.get(opt_task_id)
        if not task or not task.output_geometry_id:
            return None
        g = geom_by_id.get(task.output_geometry_id)
        if not g:
            return None
        _, coords = parse_xyz(g.xyz_data)
        return coords if coords.size else None

    def _rmsd_against(opt_task_id: int, ref: Optional[np.ndarray]) -> Optional[float]:
        if ref is None or ref.size == 0:
            return None
        coords = _coords_for(opt_task_id)
        if coords is None or coords.shape != ref.shape:
            return None
        try:
            return kabsch_rmsd(coords, ref)
        except Exception:
            return None

    def _pick_from(state: str, r: ConformerResult, ref: Optional[np.ndarray]) -> StatePick:
        return StatePick(
            state=state,
            conformer_index=r.conformer_index,
            opt_task_id=r.opt_task_id,
            rmsd_to_s0=(_rmsd_against(r.opt_task_id, ref) if state != "S0" else None),
            e_sp=r.e_singlepoint,
            e_correction=r.e_correction,
            e_combined=r.e_combined,
            e_vert_ox=r.e_vert_ox,
            e_vert_red=r.e_vert_red,
            e_vert_spin_change=r.e_vert_spin_change,
        )

    out_mols = []
    for m in mols:
        mol_id = m.id
        per_state: dict[str, list[ConformerResult]] = {}
        for (mid, st_desc), rs in groups.items():
            if mid != mol_id:
                continue
            per_state[st_desc] = rs

        if "S0" not in per_state:
            out_mols.append({
                "id": mol_id, "smiles": m.smiles, "modes": None,
                "note": "No S0 conformers extracted yet — analysis skipped.",
            })
            continue

        s0_lowest = min(
            (r for r in per_state["S0"] if r.e_combined is not None),
            key=lambda r: r.e_combined,
            default=None,
        )
        if s0_lowest is None:
            out_mols.append({
                "id": mol_id, "smiles": m.smiles, "modes": None,
                "note": "S0 has no conformer with both SP energy and free-energy correction.",
            })
            continue

        s0_ref_coords = _coords_for(s0_lowest.opt_task_id)

        # Mode 1 — lowest-energy conformer per state
        picks_low: dict[str, StatePick] = {}
        for st, pool in per_state.items():
            cand = min(
                (r for r in pool if r.e_combined is not None),
                key=lambda r: r.e_combined,
                default=None,
            )
            if cand is None:
                continue
            picks_low[st] = _pick_from(st, cand, s0_ref_coords)
        ana_low = _compute_derived(picks_low, solvation_mecn)
        ana_low.mode = "lowest_energy"

        # Mode 2 — RMSD-matched conformer per state (S0 keeps its lowest)
        picks_rmsd: dict[str, StatePick] = {}
        if "S0" in picks_low:
            picks_rmsd["S0"] = picks_low["S0"]
        for st, pool in per_state.items():
            if st == "S0":
                continue
            scored: list[tuple[float, ConformerResult]] = []
            for r in pool:
                if r.e_combined is None:
                    continue
                rmsd = _rmsd_against(r.opt_task_id, s0_ref_coords)
                if rmsd is None:
                    continue
                scored.append((rmsd, r))
            if scored:
                scored.sort(key=lambda x: x[0])
                picks_rmsd[st] = _pick_from(st, scored[0][1], s0_ref_coords)
            elif st in picks_low:
                # RMSD couldn't be computed (e.g. missing geom) — fall back
                # to the lowest-energy pick so derived quantities still
                # populate. Tagged via rmsd_to_s0 = None.
                picks_rmsd[st] = picks_low[st]
        ana_rmsd = _compute_derived(picks_rmsd, solvation_mecn)
        ana_rmsd.mode = "rmsd_matched"

        out_mols.append({
            "id": mol_id,
            "smiles": m.smiles,
            "modes": {
                "lowest_energy": _ana_to_dict(ana_low),
                "rmsd_matched":  _ana_to_dict(ana_rmsd),
            },
        })

    return {
        "project": project_name,
        "solvation_mecn": solvation_mecn,
        "sce_ref": SCE_ABS_MECN,
        "molecules": out_mols,
    }


def _ana_to_dict(a: ModeAnalysis) -> dict:
    return {
        "mode": a.mode,
        "picks": {
            st: {
                "state": p.state,
                "conformer_index": p.conformer_index,
                "opt_task_id": p.opt_task_id,
                "rmsd_to_s0": p.rmsd_to_s0,
                "e_sp": p.e_sp,
                "e_correction": p.e_correction,
                "e_combined": p.e_combined,
                "e_vert_ox": p.e_vert_ox,
                "e_vert_red": p.e_vert_red,
                "e_vert_spin_change": p.e_vert_spin_change,
            }
            for st, p in a.picks.items()
        },
        "triplet_energy_ev": a.triplet_energy_ev,
        "lambda_t1_ev": a.lambda_t1_ev,
        "dG_ox_ev": a.dG_ox_ev,
        "dG_red_ev": a.dG_red_ev,
        "E_ox_vs_sce": a.E_ox_vs_sce,
        "E_red_vs_sce": a.E_red_vs_sce,
        "lambda_ox_ev": a.lambda_ox_ev,
        "lambda_red_ev": a.lambda_red_ev,
    }
