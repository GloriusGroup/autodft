"""Submit SMILES to AutoDFT directly from Python.

Bypasses the CLI and the REST API: writes ``CalculationEntrypoint`` rows
straight into the SQLite database that the controller polls. The same
options that the dashboard form and REST endpoint expose are available
here through the ``request_metadata`` JSON field.

Run with the project root on PYTHONPATH and the production config
visible (or set ``AUTODFT_*`` env vars to override pieces of it):

    python examples/01_submit_via_python.py
"""

from __future__ import annotations

import json
from pathlib import Path

from autodft.config import load_settings
from autodft.db import get_session, init_db
from autodft.engine.entrypoint_processor import validate_smiles
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.header import ComputationHeader
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,    # GOAT GFN2-xTB
    DEFAULT_HEADER_OPTIMIZATION,  # wB97X-D3 / def2-TZVP
    DEFAULT_HEADER_SINGLEPOINT,   # wB97X-D3 / def2-QZVPD
    GXTB_HEADER_CONFSEARCH,       # GOAT g-xTB variant
)
from sqlmodel import select

# ---------------------------------------------------------------------------
# Pick up the production config (you can also rely on AUTODFT_* env vars
# alone by passing config_path=None here).
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"

settings = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
init_db(settings)  # creates data_path / comp_data / export_data / DB tables


# ---------------------------------------------------------------------------
# Helper: build the request_metadata JSON the controller understands.
# Every option in /api/submit is also a key here. The names match exactly.
# ---------------------------------------------------------------------------


def make_metadata(
    project: str,
    *,
    # Which states to build beyond S0
    request_t1: bool = False,
    request_ox: bool = False,
    request_red: bool = False,
    # Workflow toggles
    skip_confsearch: bool = False,           # GOAT skipped, RDKit geom -> optimization
    request_optimization: bool = True,        # turn off to stop after confsearch
    request_singlepoint: bool = True,         # turn off to stop after optimization
    request_singlepoint_vertical_excitations: bool = True,  # vert-ox/red/spin-flip SPs
    request_singlepoint_nbo: bool = False,    # reserved; not exposed via REST yet
    # Per-state conformer caps (defaults match the dashboard form)
    max_conformers_S0: int = 1,
    max_conformers_T1: int = 1,
    max_conformers_ox: int = 1,
    max_conformers_red: int = 1,
) -> str:
    return json.dumps({
        "project_name": project,
        "project_author": "python_example",
        # S1 is not yet supported; the controller ignores it but the
        # field is kept for forward-compatibility with the legacy schema.
        "request_S1":  False,
        "request_T1":  request_t1,
        "request_ox":  request_ox,
        "request_red": request_red,
        "request_confsearch": not skip_confsearch,
        "request_optimization": request_optimization,
        "request_singlepoint": request_singlepoint,
        "request_singlepoint_vertical_excitations": request_singlepoint_vertical_excitations,
        "request_singlepoint_nbo": request_singlepoint_nbo,
        "max_conformers_S0":  max_conformers_S0,
        "max_conformers_T1":  max_conformers_T1,
        "max_conformers_ox":  max_conformers_ox,
        "max_conformers_red": max_conformers_red,
    })


# ---------------------------------------------------------------------------
# 1) Minimal submission — S0 only, default headers, 1 conformer.
# ---------------------------------------------------------------------------

print("--- 1) minimal: ethanol, S0 only, all defaults ---")
with get_session() as session:
    entry = CalculationEntrypoint(
        smiles="CCO",
        request_metadata=make_metadata("alcohols"),
        priority=10,
        header_confsearch=DEFAULT_HEADER_CONFSEARCH,
        header_optimization=DEFAULT_HEADER_OPTIMIZATION,
        header_singlepoint=DEFAULT_HEADER_SINGLEPOINT,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    print(f"    queued #{entry.id}  {entry.smiles}")


# ---------------------------------------------------------------------------
# 2) "Everything on": T1 / ox / red + per-state conformer counts +
#    vertical excitations off + non-default priority + g-xTB conformer
#    search header.
# ---------------------------------------------------------------------------

print("--- 2) full coverage: T1/ox/red, per-state conformers, g-xTB confsearch ---")
with get_session() as session:
    entry = CalculationEntrypoint(
        smiles="c1ccc(O)cc1",  # phenol
        request_metadata=make_metadata(
            project="phenols",
            request_t1=True,
            request_ox=True,
            request_red=True,
            request_singlepoint_vertical_excitations=False,
            max_conformers_S0=5,
            max_conformers_T1=3,
            max_conformers_ox=2,
            max_conformers_red=2,
        ),
        priority=20,
        header_confsearch=GXTB_HEADER_CONFSEARCH,
        header_optimization=DEFAULT_HEADER_OPTIMIZATION,
        header_singlepoint=DEFAULT_HEADER_SINGLEPOINT,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    print(f"    queued #{entry.id}  {entry.smiles}")


# ---------------------------------------------------------------------------
# 3) Skip-confsearch path — RDKit's initial geometry feeds optimization
#    directly. Useful for very small molecules where GOAT is overkill.
# ---------------------------------------------------------------------------

print("--- 3) skip-confsearch path ---")
with get_session() as session:
    entry = CalculationEntrypoint(
        smiles="CC",  # ethane
        request_metadata=make_metadata("quick", skip_confsearch=True),
        priority=5,
        # No confsearch header needed when skip_confsearch=True
        header_confsearch=None,
        header_optimization=DEFAULT_HEADER_OPTIMIZATION,
        header_singlepoint=DEFAULT_HEADER_SINGLEPOINT,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    print(f"    queued #{entry.id}  {entry.smiles}  (skip_confsearch)")


# ---------------------------------------------------------------------------
# 4) Use a stored DB header by reading it back from the seeded rows
#    rather than hardcoding the text. This is what the dashboard does.
# ---------------------------------------------------------------------------

print("--- 4) submit using stored headers (by description lookup) ---")
with get_session() as session:
    b3lyp_opt = session.exec(
        select(ComputationHeader).where(
            ComputationHeader.kind == "optimization",
            ComputationHeader.description.contains("B3LYP")  # type: ignore[union-attr]
        )
    ).first()
    b3lyp_sp = session.exec(
        select(ComputationHeader).where(
            ComputationHeader.kind == "singlepoint",
            ComputationHeader.description.contains("B3LYP")  # type: ignore[union-attr]
        )
    ).first()
    if b3lyp_opt and b3lyp_sp:
        entry = CalculationEntrypoint(
            smiles="CCN",  # ethylamine
            request_metadata=make_metadata("amines", request_t1=True),
            priority=10,
            header_confsearch=DEFAULT_HEADER_CONFSEARCH,
            header_optimization=b3lyp_opt.header_text,
            header_singlepoint=b3lyp_sp.header_text,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        print(f"    queued #{entry.id}  {entry.smiles}  "
              f"(opt header #{b3lyp_opt.id}, sp header #{b3lyp_sp.id})")
    else:
        print("    (no B3LYP headers found — run autodft admin init-db first)")


# ---------------------------------------------------------------------------
# 5) Always validate before queuing. The controller will refuse to expand
#    bad SMILES, but failing here is faster and clearer.
# ---------------------------------------------------------------------------

print("--- 5) pre-flight validation ---")
for smi in ["c1ccccc1", "[Fe+2]", "not a smiles"]:
    v = validate_smiles(smi)
    if v["valid"]:
        print(f"    OK    {smi!r:<25}  {v['heavy_atoms']} heavy atoms, canonical={v['canonical']!r}")
    else:
        print(f"    SKIP  {smi!r:<25}  -> {v['error']}")
