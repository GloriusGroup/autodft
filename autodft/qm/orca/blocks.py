"""Calculation-specific blocks added to a user's ORCA header.

Category tasks reuse the state's singlepoint header for the method and add
only what their calculation needs, always on lines of their own.
"""

from __future__ import annotations

import re
from typing import Optional

# Default TDDFT roots for UV/Vis; enough to cover the near-UV for most organics.
UVVIS_NROOTS = 20


class HeaderConflict(ValueError):
    """The header already sets what a calculation needs to add."""


def has_block(header: str, name: str) -> bool:
    """Whether *header* has a ``%name`` block (case-insensitive)."""
    return re.search(rf"%{name}\b", header, re.IGNORECASE) is not None


def append_block(header: str, block: str) -> str:
    """*header* with *block* on new lines after it."""
    return header.rstrip("\n") + "\n" + block.rstrip("\n") + "\n"


def tddft_block(**settings) -> str:
    """A ``%tddft`` block with one ``key value`` line per setting."""
    lines = [f"  {key} {_value(value)}" for key, value in settings.items()]
    return "\n".join(["%tddft", *lines, "end"]) + "\n"


def with_tddft(header: str, **settings) -> str:
    """*header* plus a ``%tddft`` block; refuses a header that has one."""
    if has_block(header, "tddft"):
        raise HeaderConflict(
            "The singlepoint header already has a %tddft block; the UV/Vis job "
            "adds its own."
        )
    return append_block(header, tddft_block(**settings))


def compose_header(task_type: str, header: str, options: Optional[dict] = None) -> str:
    """The header a job of *task_type* runs with; other types get *header* itself.

    *options* is the task's state metadata, where per-submission settings
    such as ``uvvis_nroots`` live.
    """
    options = options or {}
    if task_type == "singlepoint_uvvis":
        return with_tddft(
            header,
            nroots=options.get("uvvis_nroots", UVVIS_NROOTS),
            tda=bool(options.get("uvvis_tda", False)),
        )
    return header


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
