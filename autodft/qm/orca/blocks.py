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
# The S1 optimisation takes root 1; a few roots above keep it converged.
S1_NROOTS = 5
_SOLVATION_RE = re.compile(r"^\s*!.*\b(?:CPCM|SMD|CPCMC)\b|%cpcm\b", re.IGNORECASE | re.MULTILINE)
# A standalone Opt: not L-Opt, COpt or the like.
_OPT_RE = re.compile(r"(?<![\w-])Opt(?![\w-])", re.IGNORECASE)
_OPT_LEVEL_RE = re.compile(r"\b(?:Tight|VeryTight|Loose|Normal|Sloppy)Opt\b", re.IGNORECASE)
_GRID_RE = re.compile(r"\bDEFGRID\d\b", re.IGNORECASE)
_KEEPDENS_RE = re.compile(r"\s*\bKeepDens\b", re.IGNORECASE)
# High-accuracy NMR (ORCA 6.1 NMR tutorial): NoFrozenCore, as core electrons shape shieldings.
NMR_DH_ROUTE = "!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF"
_SOLVENT_TOKEN_RE = re.compile(r"(?:CPCMC?|SMD)(?:\(.*\))?", re.IGNORECASE)
_DH_KEPT_BLOCKS_RE = re.compile(r"%(?:cpcm|pal)\b.*?\bend\b|%maxcore\s+\S+", re.IGNORECASE | re.DOTALL)


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


def nmr_dh_header(header: str) -> str:
    """The double-hybrid NMR route with *header*'s solvation, %maxcore and %pal."""
    solvent = [token for line in _keyword_lines(header)
               for token in line.lstrip()[1:].split() if _SOLVENT_TOKEN_RE.fullmatch(token)]
    rest = "\n".join(line.partition("#")[0] for line in header.splitlines()
                     if not line.lstrip().startswith("!"))
    kept = [match.group(0) for match in _DH_KEPT_BLOCKS_RE.finditer(rest)]
    return "\n".join([" ".join([NMR_DH_ROUTE, *solvent]), *kept]) + "\n"


def _on_keyword_lines(header: str, change) -> str:
    """*header* with *change* applied to each ``!`` line, comments left alone."""
    lines = header.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("!"):
            ending = "\n" if line.endswith("\n") else ""
            body, hash_, comment = line.rstrip("\n").partition("#")
            lines[i] = change(body) + hash_ + comment + ending
    return "".join(lines)


def with_tight_opt(header: str) -> str:
    """*header* with a plain ``Opt`` made ``TightOpt``, unless it sets an opt level already."""
    if any(_OPT_LEVEL_RE.search(line) for line in _keyword_lines(header)):
        return header
    return _on_keyword_lines(header, lambda body: _OPT_RE.sub("TightOpt", body))


def rate_header(header: str) -> str:
    """*header* for an ESD rate job: DEFGRID2 and no KeepDens (written per displacement)."""
    header = _on_keyword_lines(header, lambda body: _KEEPDENS_RE.sub("", body))
    if any(_GRID_RE.search(line) for line in _keyword_lines(header)):
        return _on_keyword_lines(header, lambda body: _GRID_RE.sub("DEFGRID2", body))
    return with_keyword(header, "DEFGRID2")


def _keyword_lines(header: str) -> list[str]:
    return [line.partition("#")[0] for line in header.splitlines() if line.lstrip().startswith("!")]


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
        if options.get("nmr_method") == "dh":
            return nmr_dh_header(header)
        return with_keyword(header, "NMR")
    if task_type == "singlepoint_soc":
        # Equilibrium solvation, as in the S1 optimisation, keeps DELE consistent.
        solvated = {"cpcmeq": True} if _SOLVATION_RE.search(header) else {}
        return with_tddft(
            header, nroots=ESD_NROOTS, iroot=1, triplets=True, dosoc=True, tda=True, **solvated,
        )
    if task_type == "optimization" and options.get("esd_role") == "S1":
        # The lowest root (Kasha), not a followed one: root following also runs during
        # the numerical frequencies and corrupts the Hessian. ORCA solvates the excited
        # state in equilibrium only for analytic gradients, so cpcmeq is explicit.
        solvated = {"cpcmeq": True} if _SOLVATION_RE.search(header) else {}
        return with_tddft(with_tight_opt(header), nroots=S1_NROOTS, iroot=1, tda=True, **solvated)
    if task_type == "optimization" and (options.get("esd_role") == "T1" or options.get(categories.ESD)):
        # ESD Hessians of T1 and of the S0 conformers (request_esd sits on S0 only).
        return with_tight_opt(header)
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
