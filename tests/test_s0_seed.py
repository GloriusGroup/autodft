"""seed_from_s0: T1, ox and red skip their conformer search and start from S0's lowest conformer."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from autodft.api import admin_ops
from autodft.engine import s0_seed
from autodft.extraction.extractor import ConformerResult, PipelineExtractor
from autodft.models import (
    ComputationHeader, ComputationTask, Molecule, MoleculeGeometry, MoleculeState, TaskStatus, TaskType,
)
from autodft.models.entrypoint import CalculationEntrypoint
from tests.test_engine import _settings

XYZ = "2\n\nC 0 0 {z}\nH 1 0 0\n"
SEEDED = ("T1", "ox", "red")


@pytest.fixture()
def seeded(session, monkeypatch):
    """S0 with two finished conformers (the second lower) and three states waiting for it."""
    header = ComputationHeader(header_text="!B3LYP Opt Freq\n")
    session.add(header)
    mol = Molecule(smiles="O=CC=O", project_name="nho/p")
    session.add(mol)
    session.commit()
    ids = dict(confsearch_header_id=header.id, optimization_header_id=header.id,
               singlepoint_header_id=header.id)
    s0 = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                       metadata_json="{}", **ids)
    states = {d: MoleculeState(molecule_id=mol.id, description=d, multiplicity=m, charge=q,
                               metadata_json=json.dumps({"seed_from_s0": True}), **ids)
              for d, m, q in (("T1", 3, 0), ("ox", 2, 1), ("red", 2, -1))}
    session.add_all([s0, *states.values()])
    session.commit()
    opts = []
    for z in (0.0, 0.1):
        opt = ComputationTask(task_type=TaskType.optimization, status=TaskStatus.successful,
                              state_id=s0.id, header_id=header.id, has_followups=False)
        session.add(opt)
        session.commit()
        geom = MoleculeGeometry(state_id=s0.id, xyz_data=XYZ.format(z=z), origin_task_id=opt.id)
        session.add(geom)
        session.commit()
        opt.output_geometry_id = geom.id
        sp = ComputationTask(task_type=TaskType.singlepoint, status=TaskStatus.successful,
                             state_id=s0.id, header_id=header.id, depends_on_task_id=opt.id,
                             has_followups=False)
        session.add_all([opt, sp])
        session.commit()
        opts.append(opt)
    energies = {opts[0].id: (-10.0, -9.90), opts[1].id: (-10.001, -9.92)}  # (E_sp, G)

    def results(self, session, mol, state):
        return [ConformerResult(molecule_id=mol.id, smiles=mol.smiles, state="S0",
                                conformer_index=i, opt_task_id=o.id,
                                e_singlepoint=energies[o.id][0], e_combined=energies[o.id][1])
                for i, o in enumerate(opts, 1)]

    monkeypatch.setattr(PipelineExtractor, "extract_state_results", results)
    return {"s0": s0, "states": states, "opts": opts, "header": header}


def _tasks(session, state):
    return session.exec(select(ComputationTask).where(ComputationTask.state_id == state.id)
                        .order_by(ComputationTask.id)).all()


def test_each_state_starts_from_the_lowest_s0_conformer(session, seeded):
    s0_seed.advance_s0_seeds(session)
    best = seeded["opts"][1]
    for state in seeded["states"].values():
        [opt] = _tasks(session, state)
        assert (opt.task_type, opt.status, opt.header_id, opt.has_followups) == (
            TaskType.optimization, TaskStatus.created, state.optimization_header_id, True)
        seed = session.get(MoleculeGeometry, opt.input_geometry_id)
        assert seed.state_id == state.id and seed.xyz_data == XYZ.format(z=0.1)
        assert seed.label == f"seed_from_task_{best.id}"
        assert json.loads(state.metadata_json)["s0_seed"] == {"task": best.id}


def test_seeding_happens_once(session, seeded):
    s0_seed.advance_s0_seeds(session)
    s0_seed.advance_s0_seeds(session)
    assert all(len(_tasks(session, s)) == 1 for s in seeded["states"].values())


@pytest.mark.parametrize("status", [TaskStatus.created, TaskStatus.pending])
def test_waits_for_every_s0_singlepoint(session, seeded, status):
    sp = [t for t in _tasks(session, seeded["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = status
    session.add(sp)
    session.commit()
    s0_seed.advance_s0_seeds(session)
    for state in seeded["states"].values():
        assert _tasks(session, state) == []
        assert "s0_seed" not in json.loads(state.metadata_json)


def test_waits_for_s0_follow_ups_not_yet_created(session, seeded):
    opt = seeded["opts"][0]
    opt.has_followups = True
    session.add(opt)
    session.commit()
    s0_seed.advance_s0_seeds(session)
    assert all(_tasks(session, s) == [] for s in seeded["states"].values())


def test_no_usable_conformer_is_recorded(session, seeded, monkeypatch):
    monkeypatch.setattr(PipelineExtractor, "extract_state_results", lambda *a: [])
    s0_seed.advance_s0_seeds(session)
    for state in seeded["states"].values():
        assert "error" in json.loads(state.metadata_json)["s0_seed"]
        assert _tasks(session, state) == []


def test_a_state_without_the_flag_is_left_alone(session, seeded):
    t1 = seeded["states"]["T1"]
    t1.metadata_json = "{}"
    session.add(t1)
    session.commit()
    s0_seed.advance_s0_seeds(session)
    assert _tasks(session, t1) == []
    assert len(_tasks(session, seeded["states"]["ox"])) == 1


def test_an_archived_molecule_is_not_seeded(session, seeded):
    mol = session.get(Molecule, seeded["s0"].molecule_id)
    mol.archived = True
    session.add(mol)
    session.commit()
    s0_seed.advance_s0_seeds(session)
    assert all(_tasks(session, s) == [] for s in seeded["states"].values())


def test_removing_s0_star_lets_the_states_seed_again(session, seeded):
    s0_seed.advance_s0_seeds(session)
    best = seeded["opts"][1]
    admin_ops._reopen_s0_seeds(session, seeded["s0"].molecule_id, {best.id})
    assert all("s0_seed" not in json.loads(s.metadata_json) for s in seeded["states"].values())


def test_removing_another_conformer_keeps_the_seed(session, seeded):
    s0_seed.advance_s0_seeds(session)
    admin_ops._reopen_s0_seeds(session, seeded["s0"].molecule_id, {seeded["opts"][0].id})
    assert all("s0_seed" in json.loads(s.metadata_json) for s in seeded["states"].values())


def test_a_rerun_of_s0_star_removes_the_seeded_optimisations(session, seeded):
    s0_seed.advance_s0_seeds(session)
    dependents, _ = admin_ops.task_dependents(session, seeded["opts"][1])
    seeded_opts = {t.id for s in seeded["states"].values() for t in _tasks(session, s)}
    assert seeded_opts <= {t.id for t in dependents}


# --- Expansion ---------------------------------------------------------------

def _expand(engine, tmp_path, monkeypatch, **metadata):
    from autodft.engine import entrypoint_processor as ep

    meta = {"project_name": "t", "request_confsearch": True, "request_T1": True,
            "request_ox": True, "request_red": True, **metadata}
    monkeypatch.setattr(ep, "_generate_initial_xyz", lambda s: "C 0 0 0\nH 1 0 0\n")
    with Session(engine) as session:
        session.add(CalculationEntrypoint(
            smiles="c1ccccc1", request_metadata=json.dumps(meta), priority=10,
            header_confsearch="!GOAT XTB2\n", header_optimization="!B3LYP OPT FREQ\n",
            header_singlepoint="!B3LYP\n",
        ))
        session.commit()
        ep.process_next_entrypoint(session, _settings(tmp_path))
        session.commit()
        states = {s.description: s for s in session.exec(select(MoleculeState)).all()}
        tasks = {d: [t.task_type for t in _tasks(session, s)] for d, s in states.items()}
        metadata = {d: json.loads(s.metadata_json) for d, s in states.items()}
    return tasks, metadata


def test_seeded_states_are_created_without_a_conformer_search(engine, tmp_path, monkeypatch):
    tasks, metadata = _expand(engine, tmp_path, monkeypatch, seed_from_s0=True, max_conformers_S0=10)
    assert tasks == {"S0": [TaskType.confsearch], "T1": [], "ox": [], "red": []}
    assert all(metadata[d]["seed_from_s0"] is True for d in SEEDED)
    assert "seed_from_s0" not in metadata["S0"] and metadata["S0"]["max_conformers_S0"] == 10


def test_without_the_flag_every_state_searches_its_own_conformers(engine, tmp_path, monkeypatch):
    tasks, metadata = _expand(engine, tmp_path, monkeypatch)
    assert tasks == {d: [TaskType.confsearch] for d in ("S0", *SEEDED)}
    assert all("seed_from_s0" not in m for m in metadata.values())


# --- Submission ----------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False])
def test_the_api_writes_the_flag_only_when_set(session, flag):
    from autodft.api.routes import SubmitRequest, _new_entrypoint

    body = SubmitRequest(smiles="c1ccccc1", request_t1=True, seed_from_s0=flag)
    entry = _new_entrypoint(session, body, "c1ccccc1", headers=("!GOAT XTB2\n", "!B3LYP OPT\n", "!B3LYP\n"))
    assert ("seed_from_s0" in json.loads(entry.request_metadata)) is flag


@pytest.mark.parametrize("flag", [True, False])
def test_the_cli_writes_the_flag_only_when_set(flag):
    from autodft.cli.submit import _build_request_metadata

    metadata = json.loads(_build_request_metadata(
        "nho/p", "nho", True, True, True, False, True, 10, 1, 1, 1, seed_from_s0=flag,
    ))
    assert ("seed_from_s0" in metadata) is flag


# --- verticals_lowest_s0 -------------------------------------------------------

VERTICALS = {TaskType.singlepoint_vert_ox, TaskType.singlepoint_vert_red, TaskType.singlepoint_vert_spin_change}


@pytest.fixture()
def lowest(session, seeded):
    """The seeded S0, flagged for verticals on S0* only."""
    s0 = seeded["s0"]
    s0.metadata_json = json.dumps({"verticals_lowest_s0": True})
    session.add(s0)
    session.commit()
    return seeded


def _verticals(session, state):
    return [t for t in _tasks(session, state) if t.task_type in VERTICALS]


@pytest.mark.parametrize("flag, expected", [(True, set()), (False, VERTICALS)])
def test_an_s0_conformer_gets_verticals_only_without_the_flag(session, seeded, flag, expected):
    from autodft.engine.state_machine import _followup_optimization

    s0, opt = seeded["s0"], seeded["opts"][0]
    _followup_optimization(session, opt, s0, {"verticals_lowest_s0": True} if flag else {})
    session.commit()
    created = {t.task_type for t in _tasks(session, s0) if t.depends_on_task_id == opt.id}
    assert created == {TaskType.singlepoint} | expected


def test_the_flag_leaves_other_states_verticals_alone(session, seeded):
    from autodft.engine.state_machine import _followup_optimization

    ox, opt = seeded["states"]["ox"], seeded["opts"][0]
    _followup_optimization(session, opt, ox, {"verticals_lowest_s0": True})
    session.commit()
    assert [t.task_type for t in _verticals(session, ox)] == [TaskType.singlepoint_vert_red]


def test_the_verticals_go_on_the_lowest_conformer_only(session, lowest):
    s0_seed.advance_s0_seeds(session)
    best = lowest["opts"][1]
    made = _verticals(session, lowest["s0"])
    assert {t.task_type for t in made} == VERTICALS and len(made) == 3
    assert all((t.depends_on_task_id, t.input_geometry_id, t.status, t.header_id)
               == (best.id, best.output_geometry_id, TaskStatus.created, lowest["s0"].singlepoint_header_id)
               for t in made)
    assert json.loads(lowest["s0"].metadata_json)["s0_verticals"] == {"task": best.id}


def test_the_verticals_are_placed_once(session, lowest):
    s0_seed.advance_s0_seeds(session)
    s0_seed.advance_s0_seeds(session)
    assert len(_verticals(session, lowest["s0"])) == 3


def test_the_verticals_wait_for_every_s0_singlepoint(session, lowest):
    sp = [t for t in _tasks(session, lowest["s0"]) if t.task_type == TaskType.singlepoint][0]
    sp.status = TaskStatus.pending
    session.add(sp)
    session.commit()
    s0_seed.advance_s0_seeds(session)
    assert _verticals(session, lowest["s0"]) == []
    assert "s0_verticals" not in json.loads(lowest["s0"].metadata_json)


def test_an_open_shell_s0_gets_no_spin_change(session, lowest):
    lowest["s0"].multiplicity = 2
    session.add(lowest["s0"])
    session.commit()
    s0_seed.advance_s0_seeds(session)
    assert {t.task_type for t in _verticals(session, lowest["s0"])} == VERTICALS - {TaskType.singlepoint_vert_spin_change}


def test_no_usable_conformer_records_why(session, lowest, monkeypatch):
    monkeypatch.setattr(PipelineExtractor, "extract_state_results", lambda *a: [])
    s0_seed.advance_s0_seeds(session)
    assert "error" in json.loads(lowest["s0"].metadata_json)["s0_verticals"]
    assert _verticals(session, lowest["s0"]) == []


def test_seeds_and_verticals_start_in_the_same_pass(session, lowest):
    s0_seed.advance_s0_seeds(session)
    assert len(_verticals(session, lowest["s0"])) == 3
    assert all(len(_tasks(session, s)) == 1 for s in lowest["states"].values())


def test_a_rerun_of_s0_star_removes_and_reopens_the_verticals(session, lowest):
    s0_seed.advance_s0_seeds(session)
    best = lowest["opts"][1]
    dependents, _ = admin_ops.task_dependents(session, best)
    assert {t.id for t in _verticals(session, lowest["s0"])} <= {t.id for t in dependents}
    admin_ops._reopen_s0_seeds(session, lowest["s0"].molecule_id, {best.id})
    assert "s0_verticals" not in json.loads(lowest["s0"].metadata_json)
    assert all("s0_seed" not in json.loads(s.metadata_json) for s in lowest["states"].values())


def test_removing_another_conformer_keeps_the_verticals(session, lowest):
    s0_seed.advance_s0_seeds(session)
    admin_ops._reopen_s0_seeds(session, lowest["s0"].molecule_id, {lowest["opts"][0].id})
    assert json.loads(lowest["s0"].metadata_json)["s0_verticals"] == {"task": lowest["opts"][1].id}


def test_only_s0_is_flagged_on_expansion(engine, tmp_path, monkeypatch):
    _, metadata = _expand(engine, tmp_path, monkeypatch, verticals_lowest_s0=True)
    assert metadata["S0"]["verticals_lowest_s0"] is True
    assert all("verticals_lowest_s0" not in metadata[d] for d in SEEDED)


@pytest.mark.parametrize("flag", [True, False])
def test_the_api_writes_the_verticals_flag_only_when_set(session, flag):
    from autodft.api.routes import SubmitRequest, _new_entrypoint

    body = SubmitRequest(smiles="c1ccccc1", verticals_lowest_s0=flag)
    entry = _new_entrypoint(session, body, "c1ccccc1", headers=("!GOAT XTB2\n", "!B3LYP OPT\n", "!B3LYP\n"))
    assert ("verticals_lowest_s0" in json.loads(entry.request_metadata)) is flag


@pytest.mark.parametrize("flag", [True, False])
def test_the_cli_writes_the_verticals_flag_only_when_set(flag):
    from autodft.cli.submit import _build_request_metadata

    metadata = json.loads(_build_request_metadata(
        "nho/p", "nho", False, True, True, False, True, 10, 1, 1, 1, verticals_lowest_s0=flag,
    ))
    assert ("verticals_lowest_s0" in metadata) is flag
