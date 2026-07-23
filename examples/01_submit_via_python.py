"""Submit SMILES to AutoDFT directly from Python.

Writes ``CalculationEntrypoint`` rows straight into the database the
controller polls — no CLI, no HTTP. The same options that the dashboard
form and REST endpoint expose are available here as keyword arguments.

Projects belong to accounts: they are stored as ``owner/project``. There
is no request here and so no API key to identify the caller, which is why
``USER`` below says whose namespace the work lands in — exactly what
``autodft submit --user`` does. Writing a bare project name would create
rows nobody owns, visible only to admin.

Run it as a script::

    python examples/01_submit_via_python.py

or import the helpers into your own code::

    from examples.submit_via_python import submit, make_metadata
"""

from __future__ import annotations

import json
from pathlib import Path

from autodft import accounts
from autodft.config import load_settings
from autodft.db import get_session, init_db
from autodft.engine.entrypoint_processor import validate_smiles
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.models.header import ComputationHeader
from autodft.qm.orca.defaults import (
    B3LYP_HEADER_OPTIMIZATION,
    B3LYP_HEADER_SINGLEPOINT,
    DEFAULT_HEADER_CONFSEARCH,
    DEFAULT_HEADER_OPTIMIZATION,
    DEFAULT_HEADER_SINGLEPOINT,
    GXTB_HEADER_CONFSEARCH,
)
from sqlmodel import select


# ---------------------------------------------------------------------------
# Configuration — edit these for your environment.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "reaction.toml"

# The account these submissions are made as. Its projects are stored as
# ``<USER>/<project>``, and only that account (plus admin) sees them.
USER = "admin"


# ---------------------------------------------------------------------------
# Engine init. ``init_db`` is idempotent: safe to call from any importing
# module; it just makes sure the data path and tables exist. It also
# creates the admin account on a fresh database, so ``USER = "admin"``
# always resolves.
# ---------------------------------------------------------------------------

SETTINGS = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
init_db(SETTINGS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def qualified_project(project: str, user: str = USER) -> tuple[str, str]:
    """Resolve a bare project name into ``(owner/project, owner)``.

    Creates the project row on first use, exactly as an API submission
    would. Two people may each have a project called ``phenols``; they are
    different projects because the owner is part of the stored name.
    """
    with get_session() as session:
        account = accounts.get_user_by_username(session, user)
        if account is None:
            raise ValueError(
                f"No account named {user!r}. Create one on the admin page "
                f"(or via POST /api/admin/users) before submitting as them."
            )
        row = accounts.get_or_create_project(session, account, project)
        return row.qualified_name, account.username


def make_metadata(
    project: str,
    author: str,
    *,
    request_t1: bool = False,
    request_ox: bool = False,
    request_red: bool = False,
    skip_confsearch: bool = False,
    request_optimization: bool = True,
    request_singlepoint: bool = True,
    request_singlepoint_vertical_excitations: bool = True,
    request_singlepoint_nbo: bool = False,
    max_conformers_S0: int = 1,
    max_conformers_T1: int = 1,
    max_conformers_ox: int = 1,
    max_conformers_red: int = 1,
) -> str:
    """Build the ``request_metadata`` JSON blob the controller consumes.

    Every keyword mirrors one body field of ``POST /api/submit``. ``S1``
    is intentionally absent — it's not supported yet.

    ``project`` must already be qualified (``owner/project``) — the
    controller matches molecules on this string verbatim. Use
    :func:`qualified_project` to build it. ``author`` is a provenance
    label; over REST the server forces it to the caller's username, so
    record the owning account here too.
    """
    return json.dumps({
        "project_name": project,
        "project_author": author,
        "request_S1": False,
        "request_T1": request_t1,
        "request_ox": request_ox,
        "request_red": request_red,
        "request_confsearch": not skip_confsearch,
        "request_optimization": request_optimization,
        "request_singlepoint": request_singlepoint,
        "request_singlepoint_vertical_excitations": request_singlepoint_vertical_excitations,
        "request_singlepoint_nbo": request_singlepoint_nbo,
        "max_conformers_S0": max_conformers_S0,
        "max_conformers_T1": max_conformers_T1,
        "max_conformers_ox": max_conformers_ox,
        "max_conformers_red": max_conformers_red,
    })


def submit(
    smiles: str,
    project: str,
    *,
    user: str = USER,
    priority: int = 10,
    header_confsearch: str | None = DEFAULT_HEADER_CONFSEARCH,
    header_optimization: str | None = DEFAULT_HEADER_OPTIMIZATION,
    header_singlepoint: str | None = DEFAULT_HEADER_SINGLEPOINT,
    **metadata_kwargs,
) -> int:
    """Validate, then queue one entrypoint. Returns the new row id.

    ``project`` is the bare name; it is qualified with *user*'s namespace
    before the row is written, the same way ``POST /api/submit`` qualifies
    it with the caller's.

    Raises ``ValueError`` if the SMILES is invalid — same check the
    REST endpoint runs before writing the row.
    """
    check = validate_smiles(smiles)
    if not check["valid"]:
        raise ValueError(f"Invalid SMILES {smiles!r}: {check['error']}")

    qualified, author = qualified_project(project, user)

    with get_session() as session:
        entry = CalculationEntrypoint(
            smiles=smiles,
            request_metadata=make_metadata(qualified, author, **metadata_kwargs),
            priority=priority,
            header_confsearch=header_confsearch,
            header_optimization=header_optimization,
            header_singlepoint=header_singlepoint,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry.id


def header_by_description(kind: str, contains: str) -> ComputationHeader | None:
    """Find a stored header by case-insensitive description substring."""
    with get_session() as session:
        return session.exec(
            select(ComputationHeader).where(
                ComputationHeader.kind == kind,
                ComputationHeader.description.contains(contains),  # type: ignore[union-attr]
            )
        ).first()


# ---------------------------------------------------------------------------
# Examples — run when this file is executed directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 1) Minimal: S0 only, defaults everywhere. "alcohols" is bare here
    #    and lands as "<USER>/alcohols" — that is the name the dashboard,
    #    the extractor and the export directory all use.
    eid = submit("CCO", project="alcohols")
    print(f"#{eid}  CCO  (defaults, project={USER}/alcohols)")

    # 2) Full coverage: T1 / ox / red, per-state conformer caps,
    #    vertical excitations off, non-default priority, g-xTB
    #    conformer-search header.
    eid = submit(
        "c1ccc(O)cc1",
        project="phenols",
        priority=20,
        request_t1=True,
        request_ox=True,
        request_red=True,
        request_singlepoint_vertical_excitations=False,
        max_conformers_S0=5,
        max_conformers_T1=3,
        max_conformers_ox=2,
        max_conformers_red=2,
        header_confsearch=GXTB_HEADER_CONFSEARCH,
    )
    print(f"#{eid}  c1ccc(O)cc1  (T1/ox/red, per-state confs, g-xTB)")

    # 3) Skip-confsearch path — RDKit's geometry feeds optimization directly.
    eid = submit(
        "CC",
        project="quick",
        priority=5,
        skip_confsearch=True,
        header_confsearch=None,
    )
    print(f"#{eid}  CC   (skip_confsearch)")

    # 4) Pick stored headers from the seeded rows by description.
    b3lyp_opt = header_by_description("optimization", "B3LYP")
    b3lyp_sp = header_by_description("singlepoint", "B3LYP")
    if b3lyp_opt and b3lyp_sp:
        eid = submit(
            "CCN",
            project="amines",
            request_t1=True,
            header_optimization=b3lyp_opt.header_text,
            header_singlepoint=b3lyp_sp.header_text,
        )
        print(f"#{eid}  CCN  (B3LYP opt #{b3lyp_opt.id} + sp #{b3lyp_sp.id})")

    # 5) Pre-flight validation — same check the REST endpoint runs.
    for smi in ["c1ccccc1", "[Fe+2]", "not a smiles"]:
        v = validate_smiles(smi)
        verdict = "OK  " if v["valid"] else "BAD "
        info = v["canonical"] if v["valid"] else v["error"]
        print(f"{verdict} {smi!r:<25}  {info}")
