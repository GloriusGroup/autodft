"""Advanced method settings as identity: existing molecules, Extend and NMR references."""

from __future__ import annotations

import json
from unittest import mock

from sqlmodel import select

from autodft.config import Settings
from autodft.db import get_session
from autodft.engine import entrypoint_processor as ep
from autodft.engine import nmr_references
from autodft.models import CalculationEntrypoint, ComputationHeader, Molecule, MoleculeState
from autodft.qm.orca.advanced import method_settings, resolve
from tests.test_advanced_submission import _legacy_project, _meta, _process, _stored, api  # noqa: F401
from tests.test_nmr_analysis import _detail, _glyoxal, _tms, db  # noqa: F401

BARE = {"header_confsearch": None, "header_optimization": "!B3LYP def2-SVP\n",
        "header_singlepoint": "!B3LYP def2-TZVP\n", "skip_confsearch": True}


def _submit(client, key, smiles="c1ccccc1", **body):
    return client.post("/api/submit", headers=key, json={"smiles": smiles, "project": "p", **body})


def _states_of(smiles):
    with get_session() as session:
        return {s.description: json.loads(s.metadata_json) for s in session.exec(
            select(MoleculeState).join(Molecule, Molecule.id == MoleculeState.molecule_id)
            .where(Molecule.smiles == smiles, Molecule.project_name == "nho/p"))}


def _bare_legacy_project(settings):
    """S0 states on a method-only header, stored without ``advanced`` (the expansion now refuses those)."""
    with get_session() as session:
        session.add(CalculationEntrypoint(
            smiles="CCO", request_metadata=json.dumps({"project_name": "nho/p", "project_author": "nho",
                                                       "request_confsearch": False}),
            header_optimization=BARE["header_optimization"], header_singlepoint=BARE["header_singlepoint"]))
        session.commit()
    with mock.patch.object(ep, "_legacy_header_error", lambda *args: None):
        _process(settings)


def test_resubmit_with_other_grid_refused(api):
    client, key, settings = api
    assert _submit(client, key).status_code == 200
    _process(settings)
    needle = "advanced.singlepoint.grid is DEFGRID3 for the existing molecule, DEFGRID2 requested"
    r = _submit(client, key, advanced={"singlepoint": {"grid": "DEFGRID2"}})
    assert r.status_code == 400 and needle in r.json()["detail"]
    r = client.post("/api/submit-batch", headers=key, json={
        "smiles_list": ["c1ccccc1"], "project": "p", "advanced": {"singlepoint": {"grid": "DEFGRID2"}}})
    assert r.json()["queued"] == [] and needle in r.json()["rejected"][0]["detail"]
    # The expansion refuses it too (the CLI and direct inserts skip the API).
    with get_session() as session:
        entry = CalculationEntrypoint(smiles="c1ccccc1", header_optimization="!wB97X-D3 def2-SVP\n",
                                      header_singlepoint="!wB97X-D3 def2-TZVP\n",
                                      header_confsearch=_confsearch_header(), request_metadata=json.dumps(
                                          {"project_name": "nho/p", "advanced": {"singlepoint": {"grid": "DEFGRID2"}}}))
        session.add(entry)
        session.commit()
        entry_id = entry.id
    _process(settings)
    with get_session() as session:
        assert needle in session.get(CalculationEntrypoint, entry_id).processing_error


def test_resubmit_of_a_legacy_molecule_in_new_mode_at_its_headers_refused(api):
    client, key, settings = api
    _bare_legacy_project(settings)
    r = _submit(client, key, smiles="CCO", **BARE)
    assert r.status_code == 400 and "header-embedded" in r.json()["detail"]


def _s0_states(smiles):
    with get_session() as session:
        return session.exec(select(MoleculeState).join(Molecule, Molecule.id == MoleculeState.molecule_id).where(
            Molecule.smiles == smiles, MoleculeState.description == "S0")).all()


def test_resubmit_with_other_grid_at_other_headers_makes_new_states(api):
    client, key, settings = api
    assert _submit(client, key).status_code == 200
    _process(settings)
    _stored("!wB97X-D3 def2-QZVP\n")
    r = _submit(client, key, header_singlepoint="!wB97X-D3 def2-QZVP\n", advanced={"singlepoint": {"grid": "DEFGRID2"}})
    assert r.status_code == 200, r.text
    _process(settings)
    grids = sorted(json.loads(s.metadata_json)["advanced"]["singlepoint"]["grid"] for s in _s0_states("c1ccccc1"))
    assert grids == ["DEFGRID2", "DEFGRID3"]


def test_resubmit_of_a_legacy_molecule_in_new_mode_at_other_headers_makes_new_states(api):
    client, key, settings = api
    _legacy_project(settings)
    r = _submit(client, key, smiles="CCO", advanced={})
    assert r.status_code == 200, r.text
    _process(settings)
    with get_session() as session:
        assert all(e.processing_error is None for e in session.exec(select(CalculationEntrypoint)))
    assert sorted("advanced" in json.loads(s.metadata_json) for s in _s0_states("CCO")) == [False, True]


def test_resubmit_with_other_nprocs_allowed(api):
    client, key, settings = api
    assert _submit(client, key).status_code == 200
    _process(settings)
    r = _submit(client, key, advanced={"optimization": {"nprocs": 16}, "singlepoint": {"maxcore": 3000}})
    assert r.status_code == 200, r.text
    _process(settings)
    with get_session() as session:
        assert all(e.processing_error is None for e in session.exec(select(CalculationEntrypoint)))
        assert len(session.exec(select(MoleculeState)).all()) == 1


def test_extend_inherits_s0_advanced(api):
    client, key, settings = api
    given = {"singlepoint": {"grid": "DEFGRID2"}, "optimization": {"nprocs": 16}}
    assert _submit(client, key, advanced=given).status_code == 200
    _process(settings)
    s0 = _states_of("c1ccccc1")["S0"]["advanced"]
    assert s0 == resolve(given, settings)[0]
    assert client.get("/api/projects/p/extend-info", headers=key).json()["advanced"] == s0
    r = client.post("/api/projects/p/extend", headers=key, json={"request_t1": True, "request_spec_ir": True})
    assert r.status_code == 200 and r.json()["rejected"] == [], r.text
    assert _meta(r.json()["queued"][0]["id"])["advanced"] == s0
    _process(settings)
    assert _states_of("c1ccccc1")["T1"]["advanced"] == s0


def _confsearch_header():
    with get_session() as session:
        state = session.exec(select(MoleculeState)).first()
        return session.get(ComputationHeader, state.confsearch_header_id).header_text


def test_extension_entrypoint_takes_the_s0_advanced(api):
    client, key, settings = api
    assert _submit(client, key, advanced={"singlepoint": {"grid": "DEFGRID2"}}).status_code == 200
    _process(settings)
    s0 = _states_of("c1ccccc1")["S0"]["advanced"]
    with get_session() as session:
        session.add(CalculationEntrypoint(
            smiles="c1ccccc1", header_optimization="!wB97X-D3 def2-SVP\n",
            header_singlepoint="!wB97X-D3 def2-TZVP\n", header_confsearch=_confsearch_header(),
            request_metadata=json.dumps({"project_name": "nho/p", "extend": True, "request_T1": True,
                                         "advanced": {}})))
        session.commit()
    _process(settings)
    assert _states_of("c1ccccc1")["T1"]["advanced"] == s0


def test_extend_legacy_project_stays_legacy(api):
    client, key, settings = api
    _legacy_project(settings)
    assert client.get("/api/projects/p/extend-info", headers=key).json()["advanced"] is None
    r = client.post("/api/projects/p/extend", headers=key, json={"request_t1": True})
    assert r.status_code == 200 and len(r.json()["queued"]) == 1, r.text
    assert "advanced" not in _meta(r.json()["queued"][0]["id"])
    _process(settings)
    assert all("advanced" not in meta for meta in _states_of("CCO").values())


def test_extend_of_a_bare_header_legacy_project_stays_legacy(api):
    client, key, settings = api
    _bare_legacy_project(settings)
    r = client.post("/api/projects/p/extend", headers=key, json={"request_t1": True})
    assert r.status_code == 200 and len(r.json()["queued"]) == 1, r.text
    assert "advanced" not in _meta(r.json()["queued"][0]["id"])


def test_bare_header_legacy_states_do_not_make_submissions_legacy(api):
    client, key, settings = api
    _bare_legacy_project(settings)
    r = _submit(client, key, **BARE)
    assert r.status_code == 200, r.text
    assert "advanced" in _meta(r.json()["id"])


def test_extend_refused_when_s0_method_settings_differ(api):
    client, key, settings = api
    assert _submit(client, key, advanced={"singlepoint": {"grid": "DEFGRID2"}}).status_code == 200
    assert _submit(client, key, smiles="CCO").status_code == 200
    _process(settings)
    info = client.get("/api/projects/p/extend-info", headers=key).json()
    assert not info["extendable"] and "Advanced" in info["reason"]
    r = client.post("/api/projects/p/extend", headers=key, json={"request_t1": True})
    assert r.status_code == 400


def test_extend_allows_s0_differing_in_resources_only(api):
    client, key, settings = api
    assert _submit(client, key, advanced={"singlepoint": {"nprocs": 4}}).status_code == 200
    assert _submit(client, key, smiles="CCO").status_code == 200
    _process(settings)
    info = client.get("/api/projects/p/extend-info", headers=key).json()
    assert info["extendable"], info["reason"]
    r = client.post("/api/projects/p/extend", headers=key, json={"request_t1": True})
    assert r.status_code == 200 and len(r.json()["queued"]) == 2, r.text
    _process(settings)
    for smiles in ("c1ccccc1", "CCO"):
        states = _states_of(smiles)
        assert states["T1"]["advanced"] == states["S0"]["advanced"]


def _reference_entries():
    with get_session() as session:
        return [e for e in session.exec(select(CalculationEntrypoint)).all()
                if json.loads(e.request_metadata)["project_name"] == nmr_references.REFERENCE_QUALIFIED]


def test_references_at_different_solvation_side_by_side(api):
    client, key, settings = api
    gas = {"optimization": {"solvation": "omit"}, "singlepoint": {"solvation": "omit"}}
    for smiles, advanced in (("CCO", {}), ("CCN", gas), ("CCC", gas), ("CCCC", {})):
        r = _submit(client, key, smiles=smiles, request_spec_nmr=True, nmr_nuclei=["H", "C"], advanced=advanced)
        assert r.status_code == 200, r.text
        _process(settings)
    refs = _reference_entries()
    assert [e.smiles for e in refs] == [nmr_references.CHCL3] * 2
    assert all(e.processing_error is None for e in refs)
    solvations = sorted(json.loads(e.request_metadata)["advanced"]["singlepoint"]["solvation"] for e in refs)
    assert solvations == ["CPCM", "omit"]
    with get_session() as session:
        states = session.exec(select(MoleculeState).join(Molecule, Molecule.id == MoleculeState.molecule_id).where(
            Molecule.project_name == nmr_references.REFERENCE_QUALIFIED)).all()
        assert len(states) == 2
        assert len({(s.optimization_header_id, s.singlepoint_header_id) for s in states}) == 1
        assert sorted(json.loads(s.metadata_json)["advanced"]["optimization"]["solvation"] for s in states) == [
            "CPCM", "omit"]


def _set_advanced(state_filter, advanced):
    with get_session() as session:
        for state in session.exec(select(MoleculeState)).all():
            molecule = session.get(Molecule, state.molecule_id)
            if state_filter(molecule):
                meta = {k: v for k, v in json.loads(state.metadata_json).items() if k != "advanced"}
                state.metadata_json = json.dumps({**meta, **({"advanced": advanced} if advanced else {})})
                session.add(state)
        session.commit()


def test_reference_lookup_requires_equal_method_settings(db):  # noqa: F811
    full = resolve({}, Settings())[0]
    gas = resolve({"singlepoint": {"solvation": "omit"}}, Settings())[0]
    with get_session() as session:
        _glyoxal(session, db, method="singlepoint")
        _tms(session, db, method="singlepoint", done=False)
    _set_advanced(lambda m: m.project_name == "nho/p", full)
    _set_advanced(lambda m: m.project_name != "nho/p", gas)
    assert _detail()["reference"]["H"]["status"] == "missing"
    # Resources do not count.
    _set_advanced(lambda m: m.project_name != "nho/p", {**full, "singlepoint": {**full["singlepoint"], "nprocs": 2}})
    assert method_settings(full) != method_settings(gas)
    assert _detail()["reference"]["H"]["status"] == "pending"
    # A legacy molecule never takes a reference with Advanced settings.
    _set_advanced(lambda m: m.project_name == "nho/p", None)
    assert _detail()["reference"]["H"]["status"] == "missing"
