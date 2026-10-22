"""Job files of states with Advanced settings; legacy states stay byte-identical."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from autodft.analysis import state_analysis
from autodft.analysis.state_analysis import detects_mecn_solvation
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.engine import nmr_references
from autodft.engine.nmr_references import nmr_solvent
from autodft.models import ComputationHeader, Molecule, MoleculeState
from autodft.qm.orca import advanced
from autodft.qm.orca.advanced import effective_header
from tests.fixtures.advanced.generate import CASES, ESD_SETTINGS, HT_SETTINGS, render, render_all, render_rates
from tests.test_nmr_analysis import _detail, _glyoxal, _tms, db  # noqa: F401 - fixture

GOLDEN = Path(__file__).parent / "fixtures" / "advanced"
METHOD_HEADERS = {
    "confsearch": '!GOAT XTB\n%xtb\n  XTBInputString "--gxtb"\nend\n%GOAT\nMAXEN 10.0\nEND\n',
    "optimization": "!wB97X-D3 def2-SVP\n",
    "singlepoint": "!wB97X-D3 def2-TZVP\n",
}
ADVANCED = advanced.defaults(Settings())
NEW = {"advanced": ADVANCED}
SCF_TIGHT = "%scf\n  Convergence Tight\nend\n"


def _new(tmp_path, name: str, settings: dict | None = None):
    task_type, description, multiplicity, base = CASES[name]
    return render(tmp_path, task_type, description, multiplicity,
                  {**base, "advanced": settings or ADVANCED}, METHOD_HEADERS)


def _with(stage: str, **values) -> dict:
    out = copy.deepcopy(ADVANCED)
    out[stage].update(values)
    return out


def test_legacy_every_type_byte_identical(tmp_path):
    rendered = render_all(tmp_path)
    assert set(rendered) == {p.name for p in GOLDEN.iterdir() if p.suffix in (".inp", ".cmd")}
    for name, text in rendered.items():
        assert text == (GOLDEN / name).read_text(), name


def test_new_mode_singlepoint_input(tmp_path):
    inp, submit = _new(tmp_path, "singlepoint")
    assert inp == (
        "!wB97X-D3 def2-TZVP\n! RIJCOSX def2/J DEFGRID3 CPCM(MeCN)\n"
        "%maxcore 1500\n%pal nprocs 8 end\n" + SCF_TIGHT + "\n*xyzfile 0 1 input.xyz\n\n"
    )
    assert "--ntasks-per-node=8\n" in submit and "--mem=12400\n" in submit
    assert submit == (GOLDEN / "singlepoint_submit.cmd").read_text()


def test_new_mode_optimization_input(tmp_path):
    inp, submit = _new(tmp_path, "optimization")
    assert inp == (
        "!wB97X-D3 def2-SVP\n! RIJCOSX def2/J DEFGRID3 CPCM(MeCN) Opt Freq\n"
        "%maxcore 2000\n%pal nprocs 12 end\n" + SCF_TIGHT + "%geom\n  Convergence Normal\nend\n"
        "\n*xyzfile 0 1 input.xyz\n\n"
    )
    assert "--ntasks-per-node=12\n" in submit and "--mem=24600\n" in submit
    assert submit == (GOLDEN / "optimization_submit.cmd").read_text()


def test_new_mode_confsearch_resources(tmp_path):
    inp, submit = _new(tmp_path, "confsearch")
    assert inp.startswith(METHOD_HEADERS["confsearch"] + "%maxcore 500\n%pal nprocs 32 end\n\n")
    assert "--ntasks-per-node=32\n" in submit


def test_new_mode_reactivity_open_shell_one_scf_block(tmp_path):
    inp, _ = _new(tmp_path, "singlepoint_vert_ox_reactivity")
    assert len(re.findall(r"%scf\b", inp)) == 1
    scf = re.search(r"%scf\b.*?\bend\b", inp, re.DOTALL).group(0)
    assert "Convergence Tight" in scf and "STABPerform true" in scf


@pytest.mark.parametrize("name", ["optimization_esd_s1", "optimization_esd_t1", "optimization_esd_s0"])
def test_new_mode_esd_s1_floor(tmp_path, name):
    inp, _ = _new(tmp_path, name)
    assert "%geom\n  Convergence Tight\nend\n" in inp
    assert "TightOpt" not in inp and inp.count("%geom") == 1


def test_new_mode_floor_only_for_esd_optimisations(tmp_path):
    for name in ("optimization", "optimization_esd_s0_added_later"):
        inp, _ = _new(tmp_path / name, name)
        assert "%geom\n  Convergence Normal\nend\n" in inp


def test_new_mode_t1_verytight_kept(tmp_path):
    inp, _ = _new(tmp_path, "optimization_esd_t1", settings=_with("optimization", geom_convergence="VeryTight"))
    assert "%geom\n  Convergence VeryTight\nend\n" in inp
    assert inp.count("%geom") == 1


def test_new_mode_s1_cpcmeq(tmp_path):
    inp, _ = _new(tmp_path, "optimization_esd_s1")
    assert "%tddft\n  nroots 5\n  iroot 1\n  tda true\n  cpcmeq true\nend\n" in inp
    gas, _ = _new(tmp_path / "gas", "optimization_esd_s1", settings=_with("optimization", solvation="omit"))
    assert "%tddft\n  nroots 5\n  iroot 1\n  tda true\nend\n" in gas


@pytest.mark.parametrize("esd_settings", [None, HT_SETTINGS], ids=["fc", "ht"])
def test_new_mode_rate_job_defgrid2(tmp_path, esd_settings):
    jobs = render_rates(tmp_path, esd_settings or ESD_SETTINGS, NEW, METHOD_HEADERS)
    tddft = [inp for inp, _ in jobs.values() if "%tddft" in inp]
    assert tddft
    for inp in tddft:
        assert "DEFGRID2" in inp and "DEFGRID3" not in inp
        assert "CPCM(MeCN)" in inp and "%maxcore" in inp


def test_new_mode_ht_opt_level_is_a_singlepoint(tmp_path):
    jobs = render_rates(tmp_path, HT_SETTINGS, NEW, METHOD_HEADERS)
    inp, submit = jobs["esd_phosp_part1"]
    assert inp.startswith("!wB97X-D3 def2-SVP ESD(PHOSP)\n! RIJCOSX def2/J DEFGRID2 CPCM(MeCN)\n"
                          "%maxcore 2000\n%pal nprocs 12 end\n" + SCF_TIGHT)
    assert not re.search(r"\b(?:Opt|Freq)\b|%geom", inp)
    assert "--ntasks-per-node=12\n" in submit


def test_new_mode_densities_keepdens(tmp_path):
    inp, _ = _new(tmp_path, "singlepoint_nbo_densities")
    assert re.search(r"^!wB97X-D3 def2-TZVP\b.*\bKeepDens\b", inp, re.MULTILINE)
    assert inp.count("KeepDens") == 1
    plain, _ = _new(tmp_path / "plain", "singlepoint")
    assert "KeepDens" not in plain


def test_effective_header_solvent():
    meta = {"advanced": ADVANCED}
    rendered = effective_header("!M B\n", meta, "singlepoint")
    assert nmr_solvent(rendered) == "CD3CN"
    assert detects_mecn_solvation(rendered)
    assert nmr_solvent("!M B\n") is None
    gas = {"advanced": _with("singlepoint", solvation="omit")}
    assert not detects_mecn_solvation(effective_header("!M B\n", gas, "singlepoint"))


@pytest.mark.parametrize("new, solvent", [(True, "CD3CN"), (False, "CDCl3 (no tabulated solvent)")])
def test_nmr_reference_solvent_from_the_rendered_header(db, new, solvent):
    with get_session() as session:
        state = _glyoxal(session, db, reference="chcl3")
        _tms(session, db, smiles=nmr_references.CHCL3)
        header = session.get(ComputationHeader, state.singlepoint_header_id)
        header.header_text = "!B3LYP def2-SVP\n"
        state.metadata_json = json.dumps({**json.loads(state.metadata_json), **(NEW if new else {})})
        session.add_all([header, state])
        session.commit()
    assert _detail()["reference"]["H"]["delta_ref_solvent"] == solvent


@pytest.mark.parametrize("new", [True, False])
def test_state_analysis_mecn_from_the_rendered_header(tmp_path, new):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    try:
        with get_session() as session:
            header = ComputationHeader(header_text="!B3LYP def2-SVP\n")
            mol = Molecule(smiles="CC=O", project_name="nho/p")
            session.add_all([header, mol])
            session.commit()
            state = MoleculeState(molecule_id=mol.id, description="S0", multiplicity=1, charge=0,
                                  metadata_json=json.dumps(NEW if new else {}),
                                  optimization_header_id=header.id, singlepoint_header_id=header.id)
            session.add(state)
            session.commit()
        assert state_analysis.analyze_project("nho/p", use_cache=False)["solvation_mecn"] is new
    finally:
        reset_engine()
