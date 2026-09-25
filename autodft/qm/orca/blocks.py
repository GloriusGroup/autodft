"""Calculation-specific blocks added to a user's ORCA header.

Category tasks reuse the state's singlepoint header for the method and add
only what their calculation needs, always on lines of their own.
"""

from __future__ import annotations

import re
from typing import Optional

from autodft import categories

# ESD TDDFT jobs: ten singlets and triplets reach past the S1 -> Tn window.
ESD_NROOTS = 10
# The S1 optimisation follows root 1; a few roots above keep the tracking stable.
S1_NROOTS = 5
# An orbital basis token of a route line; aux bases (def2/J, AutoAux) never match.
_ORBITAL_BASIS_RE = re.compile(
    r"(?:(?:ma|old|ZORA|DKH|SARC)-)*def2?-\S+|SARC-\S+|(?:[a-z]+-)?cc-p\S+|\d-\d+\+*G\S*"
    r"|STO-\dG|(?:aug-)?pc(?:S?seg|[JHX])?-\d\S*|x2c-\S+|ANO-\S+|IGLO-\S+|EPR-\S+"
    r"|SV\(P\)|SVP|TZVPP?|QZVPP?",
    re.IGNORECASE,
)
_SOLVATION_RE = re.compile(r"^\s*!.*\b(?:CPCM|SMD|CPCMC)\b|%cpcm\b", re.IGNORECASE | re.MULTILINE)


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
    """*header* plus a ``%tddft`` block; refuses a header that has one (or a %cis)."""
    if has_block(header, "tddft") or has_block(header, "cis"):
        raise HeaderConflict(
            "The header already has a %tddft (or %cis) block; this job adds "
            "its own."
        )
    return append_block(header, tddft_block(**settings))


def has_keyword(header: str, keyword: str) -> bool:
    """Whether a ``!`` line of *header* carries *keyword* (case-insensitive)."""
    pattern = rf"^\s*!.*\b{re.escape(keyword)}\b"
    return re.search(pattern, header, re.IGNORECASE | re.MULTILINE) is not None


def with_keyword(header: str, keyword: str) -> str:
    """*header* with *keyword* appended to its first ``!`` line."""
    if has_keyword(header, keyword):
        raise HeaderConflict(
            f"The singlepoint header already has the {keyword} keyword; the job adds it itself."
        )
    lines = header.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("!"):
            ending = "\n" if line.endswith("\n") else ""
            body, hash_, comment = line.rstrip("\n").partition("#")
            lines[i] = body.rstrip() + f" {keyword}" + (f" {hash_}{comment}" if hash_ else "") + ending
            return "".join(lines)
    return f"! {keyword}\n" + header


def with_basis(header: str, basis: str) -> str:
    """*header* with the orbital basis on its ``!`` lines replaced by *basis* (added if none)."""
    found = False

    def swap(match: re.Match) -> str:
        nonlocal found
        space, token = match.groups()
        if "/" in token or not _ORBITAL_BASIS_RE.fullmatch(token):
            return match.group(0)
        if found:
            return ""
        found = True
        return space + basis

    lines = header.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("!"):
            body, hash_, comment = line.partition("#")
            lines[i] = re.sub(r"(\s*)([^\s!]+)", swap, body) + hash_ + comment
    return "".join(lines) if found else with_keyword(header, basis)


def nbo_block(keywords: str) -> str:
    """A ``%nbo`` block requesting *keywords* on top of the NPA summary, plus the archive."""
    # A keyword list replaces ORCA's default one, and with it ARCHIVE (the FILE.47).
    if not re.search(r"\bARCHIVE\b", keywords, re.IGNORECASE):
        keywords = f"{keywords} ARCHIVE"
    return f'%nbo\n  NBOKEYLIST = "$NBO {keywords} $END"\nend\n'


def with_nbo(header: str, keywords: str) -> str:
    """*header* with the NBO keyword, plus a keyword-list block when *keywords*."""
    if has_block(header, "nbo"):
        raise HeaderConflict(
            "The header already has a %nbo block; this job adds its own."
        )
    header = with_keyword(header, "NBO")
    if keywords:
        header = append_block(header, nbo_block(keywords))
    return header


def densities_block(
    grid, eldens_file: str, spindens_file: str, *, eldens: bool, spindens: bool,
) -> Optional[str]:
    """A ``%plots`` block for the requested cubes, or None when neither applies."""
    if not (eldens or spindens):
        return None
    lines = [f"  dim1 {grid}", f"  dim2 {grid}", f"  dim3 {grid}", "  Format Gaussian_Cube"]
    if eldens:
        lines.append(f'  ElDens("{eldens_file}");')
    if spindens:
        lines.append(f'  SpinDens("{spindens_file}");')
    return "\n".join(["%plots", *lines, "end"]) + "\n"


def with_densities(header: str, options: dict, multiplicity: int) -> str:
    """*header* with a ``%plots`` block for the requested cubes; unchanged if none apply."""
    if has_block(header, "plots"):
        raise HeaderConflict(
            "The header already has a %plots block; this job adds its own."
        )
    defaults = categories.OPTIONS[categories.DENSITIES]
    eldens = bool(options.get("density_eldens", defaults["density_eldens"]))
    spindens = bool(options.get("density_spindens", defaults["density_spindens"])) and multiplicity > 1
    block = densities_block(
        options.get("density_grid", defaults["density_grid"]),
        options.get("density_eldens_file", defaults["density_eldens_file"]),
        options.get("density_spindens_file", defaults["density_spindens_file"]),
        eldens=eldens, spindens=spindens,
    )
    return append_block(header, block) if block else header


def compose_header(
    task_type: str, header: str, options: Optional[dict] = None, multiplicity: int = 1,
) -> str:
    """The header a job of *task_type* runs with; other types get *header* itself.

    *options* is the task's state metadata, where per-submission settings
    such as ``uvvis_nroots`` live. *multiplicity* is the job's own, for the
    singlepoint's spin-density rule (open shell only).
    """
    options = options or {}
    if task_type == "singlepoint_uvvis":
        defaults = categories.OPTIONS[categories.UVVIS]
        return with_tddft(
            header,
            nroots=options.get("uvvis_nroots", defaults["uvvis_nroots"]),
            tda=bool(options.get("uvvis_tda", defaults["uvvis_tda"])),
        )
    if task_type == "singlepoint_nmr":
        return with_keyword(with_basis(header, categories.NMR_BASIS), "NMR")
    if task_type == "singlepoint_soc":
        return with_tddft(
            header, nroots=ESD_NROOTS, iroot=1, triplets=True, dosoc=True, tda=False,
        )
    if task_type == "optimization" and options.get("esd_role") == "S1":
        # ORCA solvates the excited state in equilibrium only for analytic gradients;
        # its numerical frequencies otherwise mix in non-equilibrium LR-CPCM.
        solvated = {"cpcmeq": True} if _SOLVATION_RE.search(header) else {}
        return with_tddft(header, nroots=S1_NROOTS, iroot=1, followiroot=True, tda=False,
                          **solvated)
    if task_type == "singlepoint":
        if options.get(categories.NBO):
            defaults = categories.OPTIONS[categories.NBO]
            header = with_nbo(header, options.get("nbo_keywords", defaults["nbo_keywords"]))
        if options.get(categories.DENSITIES):
            header = with_densities(header, options, multiplicity)
        return header
    return header


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
