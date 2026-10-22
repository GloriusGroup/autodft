"""Per-submission Advanced settings: choices, validation and rendering onto a method-only header."""

from __future__ import annotations

import re
from typing import Optional

from autodft.config import Settings
from autodft.qm.orca.blocks import _keyword_lines, has_block

STAGES = ("confsearch", "optimization", "singlepoint")
RESOURCE_KEYS = ("nprocs", "maxcore")
_METHOD_KEYS = ("grid", "scf_convergence", "ri", "aux_basis", "solvation", "solvent")
STAGE_KEYS: dict[str, tuple[str, ...]] = {
    "confsearch": RESOURCE_KEYS,
    "optimization": (*_METHOD_KEYS[:2], "geom_convergence", *_METHOD_KEYS[2:], *RESOURCE_KEYS),
    "singlepoint": (*_METHOD_KEYS, *RESOURCE_KEYS),
}
CHOICES: dict[str, tuple[str, ...]] = {
    "grid": ("DEFGRID1", "DEFGRID2", "DEFGRID3", "omit"),
    "scf_convergence": ("Normal", "Strong", "Tight", "VeryTight", "Extreme", "omit"),
    "geom_convergence": ("Loose", "Normal", "Tight", "VeryTight", "omit"),
    "ri": ("RIJCOSX", "RI-J", "RIJK", "NoRI", "omit"),
    "aux_basis": ("def2/J", "def2/JK", "AutoAux", "omit"),
    "solvation": ("CPCM", "SMD", "omit"),
}
_GEOM_ORDER = ("Loose", "Normal", "Tight", "VeryTight")
_SCF_BLOCK_LEVEL = {"Normal": "Medium"}
_SOLVENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9,-]*")
_SOLVENT_MAX = 40
_RANGES = {"nprocs": (1, 64), "maxcore": (250, 16000)}
_MAX_TOTAL_MB = 400000

_SCF_LEVELS = r"(?:Sloppy|Loose|Normal|Medium|Strong|Tight|VeryTight|Extreme)"
_RESOURCE_TOKENS = ((r"PAL\d+", "set the cores under Advanced instead"),)
_OWNED_TOKENS = (
    (r"DEFGRID\d", "set the grid under Advanced instead"),
    (rf"{_SCF_LEVELS}SCF", "set the SCF convergence under Advanced instead"),
    (r"RIJCOSX|RIJONX|RI-J|RIJK|NoRI|RI", "set the RI approximation under Advanced instead"),
    (r"\S+/JK?|AutoAux", "set the auxiliary basis under Advanced instead"),
    (r"(?:CPCMC?|SMD)(?:\(.*\))?", "set the solvation under Advanced instead"),
    *_RESOURCE_TOKENS,
    (r"(?:Tight|VeryTight|Loose|Normal|Sloppy)?Opt|Freq|NumFreq|AnFreq", "the pipeline adds Opt and Freq itself"),
    (r"KeepDens", "the pipeline manages densities itself"),
)
_RESOURCE_BLOCKS = (
    ("pal", "set the cores under Advanced instead"),
    ("maxcore", "set the memory under Advanced instead"),
)
_OWNED_BLOCKS = (
    *_RESOURCE_BLOCKS,
    ("geom", "set the geometry convergence under Advanced instead"),
    ("cpcm", "set the solvation under Advanced instead"),
)
_SCF_BLOCK_RE = re.compile(r"%scf\b(.*?)\bend\b", re.IGNORECASE | re.DOTALL)


def defaults(settings: Settings) -> dict:
    """The full ``advanced`` dict from config."""
    return {stage: {key: getattr(getattr(settings.advanced, stage), key) for key in STAGE_KEYS[stage]}
            for stage in STAGES}


def _check(stage: str, key: str, value) -> Optional[str]:
    field = f"advanced.{stage}.{key}"
    if key in CHOICES:
        if value not in CHOICES[key]:
            return f"{field}: {value!r} is not one of {', '.join(CHOICES[key])}"
    elif key == "solvent":
        if not (isinstance(value, str) and _SOLVENT_RE.fullmatch(value) and len(value) <= _SOLVENT_MAX):
            return (f"{field}: {value!r} is not a solvent name "
                    f"(letters, digits, ',' or '-', no spaces; up to {_SOLVENT_MAX})")
    else:
        low, high = _RANGES[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            return f"{field}: {value!r} is not an integer from {low} to {high}"
    return None


def resolve(requested: Optional[dict], settings: Settings) -> tuple[Optional[dict], Optional[str]]:
    """*requested* filled from the config defaults, or ``(None, error)``."""
    full = defaults(settings)
    if requested is None:
        return full, None
    if not isinstance(requested, dict):
        return None, "advanced: must be an object"
    for stage, given in requested.items():
        if stage not in STAGES:
            return None, f"advanced.{stage}: unknown stage, expected one of {', '.join(STAGES)}"
        if not isinstance(given, dict):
            return None, f"advanced.{stage}: must be an object"
        for key, value in given.items():
            if key not in STAGE_KEYS[stage]:
                return None, f"advanced.{stage}.{key}: unknown setting, expected one of {', '.join(STAGE_KEYS[stage])}"
            error = _check(stage, key, value)
            if error:
                return None, error
        full[stage].update(given)
    for stage in STAGES:
        total = full[stage]["nprocs"] * full[stage]["maxcore"]
        if total > _MAX_TOTAL_MB:
            return None, f"advanced.{stage}: nprocs x maxcore is {total} MB, above {_MAX_TOTAL_MB}"
    return full, None


def check_defaults(settings: Settings) -> None:
    """Raise ValueError naming the first bad ``[advanced.*]`` config value."""
    error = resolve(defaults(settings), settings)[1]
    if error:
        raise ValueError(f"config {error}")


def method_settings(advanced: dict) -> dict:
    """*advanced* without cores and memory."""
    return {stage: {k: v for k, v in values.items() if k not in RESOURCE_KEYS}
            for stage, values in advanced.items()}


def stored_method_settings(metadata: dict) -> Optional[dict]:
    """The method settings stored in *metadata*; None for a legacy state."""
    return method_settings(metadata["advanced"]) if metadata.get("advanced") else None


def stage_for(task_type: str) -> str:
    """The Advanced stage whose settings a task of *task_type* uses."""
    return task_type if task_type in ("confsearch", "optimization") else "singlepoint"


def with_geom_floor(geom: str, floor: Optional[str]) -> str:
    """*geom* raised to *floor*; ``omit`` counts as below it."""
    if floor is not None and (geom == "omit" or _GEOM_ORDER.index(geom) < _GEOM_ORDER.index(floor)):
        return floor
    return geom


def render_settings(header: str, stage_settings: dict, stage: str, *,
                    as_singlepoint: bool = False, geom_floor: Optional[str] = None) -> str:
    """*header* (method only) plus the generated ``!`` line and blocks of one stage."""
    s = stage_settings
    optimizing = stage == "optimization" and not as_singlepoint
    geom = with_geom_floor(s.get("geom_convergence", "omit"), geom_floor)
    tokens = [s[k] for k in ("ri", "aux_basis", "grid") if s.get(k, "omit") != "omit"]
    if s.get("solvation", "omit") != "omit":
        tokens.append(f"{s['solvation']}({s['solvent']})")
    if optimizing:
        # ORCA 6.1 refuses %geom Convergence VeryTight; the keyword works.
        tokens += ["VeryTightOpt" if geom == "VeryTight" else "Opt", "Freq"]
    lines = header.rstrip("\n").splitlines()
    if tokens:
        last = max((i for i, line in enumerate(lines) if line.lstrip().startswith("!")), default=-1)
        lines.insert(last + 1, "! " + " ".join(tokens))
    lines += [f"%maxcore {s['maxcore']}", f"%pal nprocs {s['nprocs']} end"]
    if s.get("scf_convergence", "omit") != "omit":
        # Block Normal is ORCA's Strong; block Medium is the NormalSCF keyword.
        level = _SCF_BLOCK_LEVEL.get(s["scf_convergence"], s["scf_convergence"])
        lines += ["%scf", f"  Convergence {level}", "end"]
    if optimizing and geom not in ("omit", "VeryTight"):
        lines += ["%geom", f"  Convergence {geom}", "end"]
    return "\n".join(lines) + "\n"


def effective_header(header_text: str, metadata: dict, stage: str) -> str:
    """*header_text* itself for legacy metadata, else rendered with the stage's settings."""
    advanced = metadata.get("advanced")
    if not advanced:
        return header_text
    return render_settings(header_text, advanced[stage], stage)


def owned_keyword(header: str, *, resources_only: bool = False) -> Optional[str]:
    """A message naming the first setting-owned (or only cores/memory) keyword in *header*, or None."""
    owned_tokens, owned_blocks = ((_RESOURCE_TOKENS, _RESOURCE_BLOCKS) if resources_only
                                  else (_OWNED_TOKENS, _OWNED_BLOCKS))
    for line in _keyword_lines(header):
        for token in line.lstrip()[1:].split():
            for pattern, what in owned_tokens:
                if re.fullmatch(pattern, token, re.IGNORECASE):
                    return f"{token} in the header: {what}"
    masked = "\n".join(line.partition("#")[0] for line in header.splitlines())
    for name, what in owned_blocks:
        if has_block(masked, name):
            return f"%{name} in the header: {what}"
    if resources_only:
        return None
    for match in _SCF_BLOCK_RE.finditer(masked):
        if re.search(r"\bConvergence\b", match.group(1), re.IGNORECASE):
            return "%scf Convergence in the header: set the SCF convergence under Advanced instead"
    return None


def stage_owned_keyword(header: str, stage: Optional[str]) -> Optional[str]:
    """:func:`owned_keyword` as submission applies it: a confsearch header only for cores and memory."""
    return owned_keyword(header, resources_only=stage == "confsearch")


def is_legacy_header(header: str, stage: Optional[str] = None) -> bool:
    """Whether *header* (of *stage*) sets what Advanced owns."""
    return stage_owned_keyword(header, stage) is not None


def header_error(headers: tuple) -> Optional[str]:
    """Why the (confsearch, optimization, singlepoint) *headers* cannot take Advanced settings, or None."""
    for stage, text in zip(STAGES, headers):
        message = text and stage_owned_keyword(text, stage)
        if message:
            return f"{message} ({stage} header)"
    return None


def for_submission(requested, headers: tuple, settings: Settings) -> tuple[Optional[dict], Optional[str]]:
    """*requested* resolved for a submission with *headers*, or ``(None, error)``."""
    full, error = resolve(requested, settings)
    error = error or header_error(headers)
    return (None, error) if error else (full, None)
