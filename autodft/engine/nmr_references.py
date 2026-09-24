"""Reference molecules for NMR chemical shifts.

A shift is sigma_ref - sigma. Each reference (TMS for 1H and 13C, CFCl3 for
19F) is computed once per method -- the requester's optimisation and
singlepoint headers -- in a project that nobody can submit to, wipe, archive
or reassign.
"""

from __future__ import annotations

from typing import Optional

from autodft.accounts import ADMIN_USERNAME

REFERENCE_PROJECT = "system_references"
REFERENCE_QUALIFIED = f"{ADMIN_USERNAME}/{REFERENCE_PROJECT}"


def is_reference_project(name: str) -> bool:
    """Whether *name* (``owner/project`` or ``owner:project``) is the reference project."""
    from autodft.paths import normalise_project_name

    return normalise_project_name(name or "") == REFERENCE_QUALIFIED


def reserved_name_error(bare: str) -> Optional[str]:
    """Why a submission may not use the bare project name *bare*, or None."""
    if (bare or "").strip() == REFERENCE_PROJECT:
        return (
            f"{REFERENCE_PROJECT!r} is reserved for the pipeline's NMR reference "
            f"molecules; pick another project name."
        )
    return None
