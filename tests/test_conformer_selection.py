"""conformer_diversity_window: keep the lowest conformer plus the N-1 most diverse within the window."""

from __future__ import annotations

import json

import pytest
import typer
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from sqlmodel import select

from autodft.engine import conformer_selection, state_machine
from autodft.models import ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType
from tests.test_s0_seed import _expand


def _conformers(smiles: str, dihedral: tuple[int, int, int, int], angles: list[list[float]]) -> list[str]:
    """XYZ blocks of *smiles* with each listed set of angles applied to the dihedrals starting at *dihedral*."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    out = []
    for values in angles:
        conf = Chem.Conformer(mol.GetConformer())
        for k, angle in enumerate(values):
            i, j, l, m = (a + k for a in dihedral)
            rdMolTransforms.SetDihedralDeg(conf, i, j, l, m, angle)
        copy = Chem.Mol(mol)
        copy.RemoveAllConformers()
        copy.AddConformer(conf)
        out.append(Chem.MolToXYZBlock(copy))
    return out


# Hexane: all-anti, a near copy of it, one gauche, two gauche.
HEXANE = _conformers("CCCCCC", (0, 1, 2, 3), [[180, 180, 180], [179, 180, 181], [60, 180, 180], [60, 180, 60]])
# Ethylbenzene: the phenyl at 90 and at -90 is one conformer (a ring flip); 0 is another.
ETHYLBENZENE = _conformers("CCc1ccccc1", (0, 1, 2, 3), [[90], [-90], [0]])


def test_the_minimum_and_the_farthest_are_kept_not_the_near_copy():
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 0.5, 1.0], 2, 3.0) == [0, 3]
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 0.5, 1.0], 3, 3.0) == [0, 2, 3]


def test_a_ring_flip_is_not_diverse():
    assert conformer_selection.diverse(ETHYLBENZENE, [0.0, 0.1, 0.2], 2, 3.0) == [0, 2]


def test_only_the_window_is_considered():
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 0.5, 4.0], 2, 3.0) == [0, 2]


def test_a_window_holding_n_or_fewer_keeps_them_all():
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 5.0, 6.0], 3, 3.0) == [0, 1]
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 0.5, 1.0], 10, 3.0) == [0, 1, 2, 3]


def test_the_minimum_is_kept_wherever_it_is_listed():
    assert conformer_selection.diverse(HEXANE, [1.0, 0.1, 0.5, 0.0], 2, 3.0) == [3, 0]


def test_n_below_one_keeps_none():
    assert conformer_selection.diverse(HEXANE, [0.0, 0.1, 0.5, 1.0], 0, 3.0) == []


def test_mismatched_geometries_are_not_comparable():
    assert conformer_selection.diverse([*HEXANE[:2], ETHYLBENZENE[0]], [0.0, 0.1, 0.2], 2, 3.0) is None


# --- The confsearch follow-up --------------------------------------------------

@pytest.fixture()
def searched(session):
    """A finished S0 conformer search with the four hexane conformers."""
    header = ComputationHeader(header_text="!B3LYP Opt\n")
    session.add(header)
    mol = Molecule(smiles="CCCCCC", project_name="nho/p")
    session.add(mol)
    session.commit()
    state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0, metadata_json="{}",
                          confsearch_header_id=header.id, optimization_header_id=header.id,
                          singlepoint_header_id=header.id)
    session.add(state)
    session.commit()
    task = ComputationTask(task_type=TaskType.confsearch, status=TaskStatus.successful,
                           state_id=state.id, header_id=header.id)
    session.add(task)
    session.commit()
    geoms = [MoleculeGeometry(state_id=state.id, xyz_data=x, energy=e, origin_task_id=task.id, label=f"conformer_{i}")
             for i, (x, e) in enumerate(zip(HEXANE, [0.0, 0.1, 0.5, 1.0]))]
    session.add_all(geoms)
    session.commit()
    return task, state, geoms


def _optimised(session, state):
    tasks = session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id,
                                                       ComputationTask.task_type == TaskType.optimization)).all()
    return sorted(session.get(MoleculeGeometry, t.input_geometry_id).label for t in tasks)


def test_without_a_window_the_lowest_are_kept(session, searched):
    task, state, _ = searched
    state_machine._followup_confsearch(session, task, state, {"max_conformers_S0": 2})
    assert _optimised(session, state) == ["conformer_0", "conformer_1"]


def test_with_a_window_the_diverse_are_kept(session, searched):
    task, state, _ = searched
    state_machine._followup_confsearch(session, task, state, {"max_conformers_S0": 2, "conformer_diversity_window": 3.0})
    assert _optimised(session, state) == ["conformer_0", "conformer_3"]


def test_unreadable_conformers_fall_back_to_the_lowest(session, searched):
    task, state, geoms = searched
    geoms[2].xyz_data = ETHYLBENZENE[0]
    session.add(geoms[2])
    session.commit()
    state_machine._followup_confsearch(session, task, state, {"max_conformers_S0": 2, "conformer_diversity_window": 3.0})
    assert _optimised(session, state) == ["conformer_0", "conformer_1"]


# --- Expansion and submission ----------------------------------------------------

def test_every_state_carries_the_window(engine, tmp_path, monkeypatch):
    _, metadata = _expand(engine, tmp_path, monkeypatch, conformer_diversity_window=3)
    assert {d: m["conformer_diversity_window"] for d, m in metadata.items()} == dict.fromkeys(metadata, 3.0)


def test_without_a_window_no_state_carries_one(engine, tmp_path, monkeypatch):
    _, metadata = _expand(engine, tmp_path, monkeypatch)
    assert all("conformer_diversity_window" not in m for m in metadata.values())


@pytest.mark.parametrize("window", [2.5, None])
def test_the_api_writes_the_window_only_when_set(session, window):
    from autodft.api.routes import SubmitRequest, _new_entrypoint

    body = SubmitRequest(smiles="c1ccccc1", conformer_diversity_window=window)
    entry = _new_entrypoint(session, body, "c1ccccc1", headers=("!GOAT XTB2\n", "!B3LYP OPT\n", "!B3LYP\n"))
    assert json.loads(entry.request_metadata).get("conformer_diversity_window") == window


@pytest.mark.parametrize("window", [0.0, -1.0, float("nan")])
def test_the_api_refuses_a_window_not_above_zero(window):
    from autodft.api.routes import SubmitRequest, _conformers_error

    response = _conformers_error(SubmitRequest(smiles="c1ccccc1", conformer_diversity_window=window))
    assert response is not None and response.status_code == 400


@pytest.mark.parametrize("window", [2.5, None])
def test_the_cli_writes_the_window_only_when_set(window):
    from autodft.cli.submit import _build_request_metadata

    metadata = json.loads(_build_request_metadata(
        "nho/p", "nho", False, True, True, False, True, 10, 1, 1, 1, conformer_diversity_window=window,
    ))
    assert metadata.get("conformer_diversity_window") == window


@pytest.mark.parametrize("window", [0.0, -1.0, float("nan")])
def test_the_cli_refuses_a_window_not_above_zero(window):
    from autodft.cli.submit import _build_request_metadata

    with pytest.raises(typer.BadParameter):
        _build_request_metadata("nho/p", "nho", False, True, True, False, True, 10, 1, 1, 1,
                                conformer_diversity_window=window)
