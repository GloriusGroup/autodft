"""Raw-file exports, cleanup and the photophysics workbook."""

from __future__ import annotations

import pytest

from autodft.extraction.extractor import _FILE_MAP, _copy_task_files


def test_legacy_file_maps_are_unchanged():
    assert _FILE_MAP["optimization"] == [
        ("input.inp", "opt_input.inp"), ("input.xyz", "opt_geometry.xyz"), ("output.out", "opt_output.out"),
    ]
    assert _FILE_MAP["singlepoint"] == [
        ("input.inp", "sp_input.inp"), ("input.xyz", "sp_geometry.xyz"), ("output.out", "sp_output.out"),
    ]
    for kind in ("vert_spin_change", "vert_ox", "vert_red"):
        assert _FILE_MAP[f"singlepoint_{kind}"] == [
            ("input.inp", f"sp_{kind}_input.inp"), ("input.xyz", f"sp_{kind}_geometry.xyz"),
            ("output.out", f"sp_{kind}_output.out"),
        ]
    assert _FILE_MAP["confsearch"] == [
        ("input.inp", "confsearch_input.inp"), ("output.out", "confsearch_output.out"),
        ("input.finalensemble.xyz", "confsearch_ensemble.xyz"),
    ]


@pytest.mark.parametrize("task_type,prefix", [
    ("singlepoint_uvvis", "sp_uvvis"), ("singlepoint_nmr", "sp_nmr"), ("singlepoint_soc", "sp_soc"),
    ("esd_isc", "esd_isc"), ("esd_risc", "esd_risc"), ("esd_ic", "esd_ic"), ("esd_fluor", "esd_fluor"),
    ("esd_isc_t1s0", "esd_isc_t1s0"), ("esd_phosp", "esd_phosp"),
])
def test_new_job_types_export_input_geometry_and_output(tmp_path, task_type, prefix):
    job = tmp_path / "job"
    job.mkdir()
    for name in ("input.inp", "input.xyz", "output.out", "input.gbw", "initial.hess"):
        (job / name).write_text(name)
    assert _copy_task_files(job, tmp_path / "out", 1, task_type) == 3
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(
        f"conf1_{prefix}_{suffix}" for suffix in ("input.inp", "geometry.xyz", "output.out")
    )
