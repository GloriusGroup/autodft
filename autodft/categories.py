"""Opt-in calculation categories beyond the S0 / T1 / ox / red states.

Each category is a ``request_metadata`` flag that only submissions made after
it existed carry. A molecule without the flag never takes its code path,
which is what keeps projects already in flight unaffected.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from sqlmodel import Session, select

ESD = "request_esd"
ESD_HT = "request_esd_ht"
UVVIS = "request_spec_uvvis"
IR = "request_spec_ir"
NMR = "request_spec_nmr"
DENSITIES = "request_densities"
NBO = "request_singlepoint_nbo"

CATEGORIES = (ESD, UVVIS, IR, NMR, DENSITIES, NBO)
LABELS = {ESD: "ESD", UVVIS: "UV/Vis", IR: "IR", NMR: "NMR", DENSITIES: "Densities", NBO: "NBO"}

# What the engine can compute today. Anything else is refused at submission
# rather than accepted and silently never run.
AVAILABLE = frozenset(CATEGORIES)

# Categories whose jobs are built on the singlepoint header.
_ON_SP_HEADER = frozenset({ESD, UVVIS, NMR})

_FREQ_RE = re.compile(r"^\s*!.*\b(?:Freq|NumFreq|AnFreq)\b", re.IGNORECASE | re.MULTILINE)
_TDDFT_RE = re.compile(r"%(?:tddft|cis)\b", re.IGNORECASE)
# ORCA 6.1 has no native third derivatives for B88-exchange functionals, which
# TDDFT gradients need; their LibXC(...) versions work (CAM-B3LYP is fine).
# A route-line token of its own, so LibXC(B3LYP) and CAM-B3LYP do not match.
# A refused header is cheaper than a wave of failed S1 optimisations tripping
# the circuit breaker.
_NATIVE_B88_RE = re.compile(
    r"^\s*!(?:.*\s)?(?:RI-)?(?:B3LYP|BLYP|BP86|BP|B3P86|B3PW91|BPW91|B1LYP|BHANDHLYP|B2PLYP"
    r"|B2GP-PLYP|X3LYP)(?![\w(])",
    re.IGNORECASE | re.MULTILINE,
)
_SP_CONFLICTS = (
    (re.compile(r"%tddft\b", re.IGNORECASE), "%tddft"),
    (re.compile(r"%cis\b", re.IGNORECASE), "%cis"),
    (re.compile(r"%eprnmr\b", re.IGNORECASE), "%eprnmr"),
    (re.compile(r"%esd\b", re.IGNORECASE), "%esd"),
    (re.compile(r"^\s*!.*\bNMR\b", re.IGNORECASE | re.MULTILINE), "the NMR keyword"),
    (re.compile(r"^\s*!.*\bESD\b", re.IGNORECASE | re.MULTILINE), "the ESD keyword"),
    (_FREQ_RE, "a frequency keyword (Freq, NumFreq, AnFreq)"),
    (re.compile(r"^\s*!.*\b\w*Opt(?:TS|H)?\b", re.IGNORECASE | re.MULTILINE), "an optimisation keyword"),
)
# Densities and NBO add their own %plots / %nbo block to the singlepoint
# header; checked only against the category that would collide with it.
_DENSITY_CONFLICT = (re.compile(r"%plots\b", re.IGNORECASE), "%plots")
_NBO_CONFLICTS = (
    (re.compile(r"^\s*!.*\bNBO\b", re.IGNORECASE | re.MULTILINE), "the NBO keyword"),
    (re.compile(r"%nbo\b", re.IGNORECASE), "%nbo"),
)
# No leading "-" (submit.cmd's `cp *.cube` would read it as an option) or
# "." (a dotfile the same glob never matches).
_CUBE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}\.cube\Z")
# No $, quotes or newlines: nbo_keywords goes inside NBOKEYLIST = "$NBO ... $END".
_NBO_KEYWORDS_RE = re.compile(r"^[A-Za-z0-9 =_.,+-]{0,200}\Z")

# Settings each category takes, with defaults. Stored -- defaults filled in --
# only when the category is requested.
OPTIONS: dict[str, dict] = {
    ESD: {"esd_tn_window_ev": 0.2, "esd_temperature_k": 298.15},
    UVVIS: {"uvvis_nroots": 20, "uvvis_tda": False},
    NMR: {"nmr_nuclei": ["H", "C", "F"]},
    DENSITIES: {
        "density_eldens": True,
        "density_spindens": True,
        "density_grid": 75,
        "density_eldens_file": "ElDens.cube",
        "density_spindens_file": "SpinDens.cube",
    },
    NBO: {"nbo_keywords": ""},
}
UVVIS_NROOTS_MAX = 100
NMR_NUCLEI = ("H", "C", "F")


def requested(metadata: dict) -> set[str]:
    """The categories *metadata* asks for."""
    return {key for key in CATEGORIES if metadata.get(key)}


def options(metadata: dict) -> dict:
    """The settings of the requested categories, defaults filled in."""
    out: dict = {}
    for key in sorted(requested(metadata)):
        for name, default in OPTIONS.get(key, {}).items():
            value = metadata.get(name, default)
            # Lists are copied so no caller can mutate a shared default.
            out[name] = list(value) if isinstance(value, list) else value
    return out


def snapshot(metadata: dict) -> dict:
    """Category keys and settings to store with a state: only for requested
    categories, so an unflagged submission's metadata stays as it was."""
    flags = {key: True for key in (*CATEGORIES, ESD_HT) if metadata.get(key)}
    return {**flags, **options(metadata)}


def every_state_snapshot(metadata: dict) -> dict:
    """Densities and NBO settings for every state's own energy singlepoint,
    not only S0's -- ``{}`` when neither is requested.

    ``request_singlepoint_nbo`` itself already flows onto every state through
    ``_create_state``'s defaults, so only its ``nbo_keywords`` option is
    added here.
    """
    out: dict = {}
    if metadata.get(DENSITIES):
        out[DENSITIES] = True
        out.update({name: metadata.get(name, default)
                    for name, default in OPTIONS[DENSITIES].items()})
    if metadata.get(NBO):
        out.update({name: metadata.get(name, default)
                    for name, default in OPTIONS[NBO].items()})
    return out


def esd_settings(metadata: dict) -> dict:
    """The HT switch and ESD options, for the S1 and T1 states whose jobs read them."""
    settings = {name: metadata.get(name, default) for name, default in OPTIONS[ESD].items()}
    return {ESD_HT: bool(metadata.get(ESD_HT)), **settings}


def on_s0(description: str, metadata: dict, key: str) -> bool:
    """Whether a state asks for the S0-only category *key*."""
    return description == "S0" and bool(metadata.get(key))


def rejection(
    check: dict,
    metadata: dict,
    header_optimization: Optional[str],
    header_singlepoint: Optional[str],
) -> Optional[str]:
    """Why the requested categories cannot run for this molecule, or None.

    *check* carries the reference's ``multiplicity``; the closed-shell rules of
    later categories read only that.
    """
    if metadata.get(ESD_HT) and not metadata.get(ESD):
        return "request_esd_ht only applies together with request_esd."

    wanted = requested(metadata)
    if not wanted:
        return None

    unavailable = sorted(LABELS[key] for key in wanted - AVAILABLE)
    if unavailable:
        return f"{', '.join(unavailable)} is not available yet."

    closed_shell = sorted(LABELS[key] for key in wanted & {ESD, NMR})
    if closed_shell and check.get("multiplicity") != 1:
        return (
            f"{', '.join(closed_shell)} needs a closed-shell singlet reference; "
            f"this molecule has multiplicity {check.get('multiplicity')}."
        )

    if not metadata.get("request_optimization", True):
        labels = ", ".join(sorted(LABELS[key] for key in wanted))
        return f"{labels} needs the optimisation stage."

    if ESD in wanted:
        if not metadata.get("request_singlepoint", True):
            return (
                "ESD needs the energy singlepoint stage; its energies pick the "
                "lowest conformer and set the rates."
            )
        if not _FREQ_RE.search(header_optimization or ""):
            return (
                "ESD needs Freq in the optimisation header; the rates are built "
                "from the S0, S1 and T1 Hessians."
            )
        if _TDDFT_RE.search(header_optimization or ""):
            return (
                "ESD adds its own %tddft block to the optimisation header for the "
                "S1 optimisation; pick an optimisation header without one."
            )
        if _NATIVE_B88_RE.search(header_optimization or ""):
            return (
                "ESD's S1 optimisation needs TDDFT gradients, which ORCA 6.1 cannot "
                "compute with a native B88 functional (B3LYP, BLYP, BP86, ...); use "
                "its LibXC version in the optimisation header, e.g. !LibXC(B3LYP)."
            )
        window = metadata.get("esd_tn_window_ev", OPTIONS[ESD]["esd_tn_window_ev"])
        if not _number(window) or not 0 <= window <= 1:
            return "The ESD triplet window must be between 0 and 1 eV (esd_tn_window_ev)."
        temperature = metadata.get("esd_temperature_k", OPTIONS[ESD]["esd_temperature_k"])
        if not _number(temperature) or not 0 < temperature <= 1000:
            return "The ESD temperature must be above 0 and at most 1000 K (esd_temperature_k)."

    if UVVIS in wanted:
        nroots = metadata.get("uvvis_nroots", OPTIONS[UVVIS]["uvvis_nroots"])
        if (isinstance(nroots, bool) or not isinstance(nroots, int)
                or not 1 <= nroots <= UVVIS_NROOTS_MAX):
            return f"UV/Vis needs between 1 and {UVVIS_NROOTS_MAX} excited states (uvvis_nroots)."

    if NMR in wanted:
        nuclei = metadata.get("nmr_nuclei", OPTIONS[NMR]["nmr_nuclei"])
        if (not isinstance(nuclei, list) or not nuclei
                or any(n not in NMR_NUCLEI for n in nuclei)):
            return (
                f"NMR nuclei must be a non-empty subset of "
                f"{', '.join(NMR_NUCLEI)} (nmr_nuclei)."
            )

    if IR in wanted and not _FREQ_RE.search(header_optimization or ""):
        return (
            "IR needs Freq in the optimisation header; its spectrum comes from "
            "the frequency calculation."
        )

    if DENSITIES in wanted:
        if not metadata.get("request_singlepoint", True):
            return (
                "Densities needs the energy singlepoint stage; the cube files "
                "come from that job."
            )
        if _DENSITY_CONFLICT[0].search(header_singlepoint or ""):
            return (
                f"The singlepoint header already contains {_DENSITY_CONFLICT[1]}. "
                f"Densities adds its own %plots block; use the tick box instead."
            )
        defaults = OPTIONS[DENSITIES]
        if not (metadata.get("density_eldens", defaults["density_eldens"])
                or metadata.get("density_spindens", defaults["density_spindens"])):
            return "Densities needs at least one of density_eldens / density_spindens."
        grid = metadata.get("density_grid", defaults["density_grid"])
        if isinstance(grid, bool) or not isinstance(grid, int) or not 10 <= grid <= 300:
            return "density_grid must be an integer between 10 and 300."
        eldens_file = metadata.get("density_eldens_file", defaults["density_eldens_file"])
        spindens_file = metadata.get("density_spindens_file", defaults["density_spindens_file"])
        if (not isinstance(eldens_file, str) or not isinstance(spindens_file, str)
                or not _CUBE_NAME_RE.match(eldens_file) or not _CUBE_NAME_RE.match(spindens_file)):
            return (
                "density_eldens_file and density_spindens_file must look like "
                "NAME.cube (letters, digits, _.- only)."
            )
        if eldens_file == spindens_file:
            return "density_eldens_file and density_spindens_file must differ."

    if NBO in wanted:
        if not metadata.get("request_singlepoint", True):
            return "NBO needs the energy singlepoint stage; it runs inside that job."
        for pattern, label in _NBO_CONFLICTS:
            if pattern.search(header_singlepoint or ""):
                return (
                    f"The singlepoint header already contains {label}. NBO adds "
                    f"its own %nbo block; use the tick box instead."
                )
        keywords = metadata.get("nbo_keywords", OPTIONS[NBO]["nbo_keywords"])
        if not isinstance(keywords, str) or not _NBO_KEYWORDS_RE.match(keywords):
            return (
                "nbo_keywords may only contain letters, digits, spaces and "
                "=_.,+- -- no $, quotes or newlines (nbo_keywords)."
            )

    if wanted & _ON_SP_HEADER:
        for pattern, label in _SP_CONFLICTS:
            if pattern.search(header_singlepoint or ""):
                return (
                    f"The singlepoint header already contains {label}. UV/Vis, "
                    f"NMR and ESD add their own blocks to it; pick a plain "
                    f"singlepoint header."
                )
    return None


def nbo_unavailable(settings) -> Optional[str]:
    """Why NBO cannot run given *settings*, or None.

    Kept apart from ``rejection`` (which is settings-free) so the API and CLI
    can check it once, against the process's own configuration.
    """
    if not settings.orca.nbo_exe:
        return "NBO is not available: [orca].nbo_exe is not configured."
    return None


def existing_conflict(
    session: Session, project_name: str, canonical_smiles: str, wanted: set[str],
    requested_options: Optional[dict] = None,
) -> Optional[str]:
    """Refuse categories for a molecule that already exists without them, or
    with different options for them.

    Categories are only added to new molecules: finished work is never
    extended, so a molecule computed before a category existed stays as it was.
    """
    if not wanted:
        return None

    from autodft.models.molecule import Molecule
    from autodft.models.state import MoleculeState

    molecule = session.exec(
        select(Molecule).where(
            Molecule.smiles == canonical_smiles,
            Molecule.project_name == project_name,
        )
    ).first()
    if molecule is None:
        return None

    have: set[str] = set()
    stored_metadata: dict = {}
    for state in session.exec(
        select(MoleculeState).where(
            MoleculeState.molecule_id == molecule.id,
            MoleculeState.description == "S0",
        )
    ).all():
        metadata = json.loads(state.metadata_json) if state.metadata_json else {}
        have |= requested(metadata)
        stored_metadata.update(metadata)

    missing = sorted(LABELS[key] for key in wanted - have)
    if missing:
        return (
            f"{canonical_smiles} already exists in {project_name} (molecule "
            f"{molecule.id}) without {', '.join(missing)}. Categories are only added "
            f"to new molecules; submit it into another project."
        )

    if requested_options is not None:
        stored_options = options(stored_metadata)

        def value(name: str, source: dict, source_metadata: dict):
            if name == ESD_HT:
                return bool(source_metadata.get(ESD_HT))
            found = source.get(name)
            # Nuclei are a set; their order means nothing.
            return sorted(found) if isinstance(found, list) else found

        for key in sorted(wanted):
            names = (*OPTIONS.get(key, {}), *((ESD_HT,) if key == ESD else ()))
            for name in names:
                old = value(name, stored_options, stored_metadata)
                new = value(name, requested_options, requested_options)
                if old != new:
                    return (
                        f"{canonical_smiles} already exists in {project_name} (molecule "
                        f"{molecule.id}) with {name}={old!r}; category options apply to new "
                        f"molecules only, so submit it into another project."
                    )
    return None


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
