"""ORCA ESD rate inputs, built from the SOC singlepoints and Hessians of S0*, S1 and T1.

Energies are TDDFT-consistent: E(S0) is the ground state at S0*, E(S1) the
S1 root at the S1 geometry, E(T_n) the T_n root at the T1 geometry (T_n for
n >= 2 borrows the T1 geometry and Hessian: the shifted-T1 proxy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from autodft import categories
from autodft.qm.orca import blocks, esd_parser
from autodft.qm.orca.esd_parser import EH_TO_CM, EV_TO_EH

# Rate task type -> (initial state, final state, Hessian file -> state).
RATES: dict[str, tuple[str, str, dict[str, str]]] = {
    "esd_isc": ("S1", "T1", {"initial.hess": "S1", "final.hess": "T1"}),
    "esd_risc": ("T1", "S1", {"initial.hess": "T1", "final.hess": "S1"}),
    "esd_ic": ("S1", "S0", {"gs.hess": "S0", "es.hess": "S1"}),
    "esd_fluor": ("S1", "S0", {"gs.hess": "S0", "es.hess": "S1"}),
    "esd_isc_t1s0": ("T1", "S0", {"initial.hess": "T1", "final.hess": "S0"}),
    "esd_phosp": ("T1", "S0", {"gs.hess": "S0", "ts.hess": "T1"}),
}
_HESSIAN_KEYS = {
    "initial.hess": "ISCISHESS", "final.hess": "ISCFSHESS",
    "gs.hess": "GSHESSIAN", "es.hess": "ESHESSIAN", "ts.hess": "TSHESSIAN",
}
SUBLEVELS = (-1, 0, 1)
# FC rates need no electronic structure: ORCA gets DELE and SOCME from us.
FC_HEADER = "! ESD(ISC) NOITER\n%maxcore 2000\n"
# Opens the first diagnostic job; the output check needs rates only from the jobs before it.
DIAGNOSTICS_MARKER = "# autodft: FC diagnostics from here on, not part of the rate\n"

FC_RATES = ("esd_isc", "esd_risc", "esd_isc_t1s0")
# TDDFT roots of a rate job: ISC types reach this many past the highest root they read.
ROOTS_ABOVE = 3
NROOTS = {"esd_fluor": 5, "esd_ic": 5, "esd_phosp": 10}
# Manual 5.5.4.2: above ~5 cm-1 of SOC the Herzberg-Teller term is negligible.
HT_SOCME_CM = 5.0
# Beyond this many kT uphill, k_RISC sits at ESD's numerical floor (M5): FC, and flagged.
RISC_UPHILL_KT = 10
KT_CM_PER_K = 0.6950348
# ISC and IC rates: an explicit Gaussian inhomogeneous width (ORCA's 250 cm-1 is wide).
LINESHAPE = {"LINES": "GAUSS", "INLINEW": "100"}
# FC reruns of every ISC-type channel, for the flags only; never part of the rate.
DIAGNOSTICS = {
    "inlinew_100": LINESHAPE,
    "inlinew_25": {"LINES": "GAUSS", "INLINEW": "25"},
    "inlinew_250": {"LINES": "GAUSS", "INLINEW": "250"},
    "tcutfreq_300": {**LINESHAPE, "TCUTFREQ": "300"},
}


def is_fc(rate: str, options: dict, computed: Optional[dict] = None) -> bool:
    """Whether *rate* runs FC jobs only: no electronic structure, seconds on one core.

    *computed* is the built task's record, whose ``ht`` is the per-channel decision.
    """
    ht = (computed or {}).get("ht", categories.esd_settings(options)[categories.ESD_HT])
    return rate in FC_RATES and not ht


def ht_decision(rate: str, socme_cm: float, dele_cm: float, settings: dict) -> tuple[bool, str]:
    """Whether a channel of *rate* gets Herzberg-Teller, and why (``categories.esd_settings``)."""
    if not settings[categories.ESD_HT]:
        return False, "Herzberg-Teller not requested"
    if rate == "esd_ic":
        return False, "no Herzberg-Teller for IC"
    if rate in ("esd_fluor", "esd_phosp"):
        return True, "always for emission"
    if rate == "esd_risc":
        limit = RISC_UPHILL_KT * KT_CM_PER_K * settings["esd_temperature_k"]
        if -dele_cm > limit:
            return False, f"{-dele_cm:.0f} cm-1 uphill > {RISC_UPHILL_KT} kT ({limit:.0f} cm-1)"
    if socme_cm >= HT_SOCME_CM:
        return False, f"SOCME {socme_cm:.2f} cm-1 >= {HT_SOCME_CM:g}"
    return True, f"SOCME {socme_cm:.2f} cm-1 < {HT_SOCME_CM:g}"


class EsdInputError(ValueError):
    """A rate job's inputs are missing or unusable."""


@dataclass(frozen=True)
class StateData:
    """What the rate jobs read from one state's SOC singlepoint."""

    ground: float
    singlets: dict[int, float]
    triplets: dict[int, float]
    socme: dict[tuple[int, int], float]

    @classmethod
    def parse(cls, content: str) -> "StateData":
        ground = esd_parser.ground_energy(content)
        singlets = esd_parser.excitation_energies(content, "SINGLETS")
        triplets = esd_parser.excitation_energies(content, "TRIPLETS")
        socme = esd_parser.socme(content)
        if ground is None or not singlets or not triplets or not socme:
            raise EsdInputError("the SOC singlepoint output lacks its roots or its SOC matrix")
        return cls(ground, singlets, triplets, socme)

    def singlet(self, n: int) -> float:
        """Total energy (Eh) of S_n at this geometry."""
        return self.ground + self.singlets[n]

    def triplet(self, n: int) -> float:
        """Total energy (Eh) of T_n at this geometry."""
        return self.ground + self.triplets[n]


def energy(state: str, data: StateData) -> float:
    """E(S0), E(S1) or E(T1) (Eh), each at its own geometry."""
    if state == "S0":
        return data.ground
    return data.singlet(1) if state == "S1" else data.triplet(1)


def isc_channels(s1: StateData, t1: StateData, window_ev: float) -> list[int]:
    """T1 always; T_n when E(T_n) <= E(S1) + *window_ev*."""
    limit = s1.singlet(1) + window_ev * EV_TO_EH
    return [n for n in sorted(t1.triplets) if n == 1 or t1.triplet(n) <= limit]


def esd_block(**settings) -> str:
    """A ``%esd`` block with one ``KEY value`` line per setting."""
    lines = [f"  {key} {value}" for key, value in settings.items()]
    return "\n".join(["%esd", *lines, "end"]) + "\n"


def build(
    rate: str, data: dict[str, StateData], options: dict, sp_header: str, charge: int,
) -> tuple[str, dict]:
    """The input of one rate task (without its last geometry line) and what went into it.

    *options* is the rate task's state metadata (``categories.esd_settings`` keys).
    """
    settings = categories.esd_settings(options)
    initial, final, hessians = RATES[rate]
    files = {_HESSIAN_KEYS[name]: f'"{name}"' for name in hessians}
    temperature = {"TEMP": f"{settings['esd_temperature_k']:g}"}
    energies = {state: energy(state, data[state]) for state in (initial, final)}
    sp_header = blocks.rate_header(sp_header)
    diagnostics: list[dict] = []

    if rate in FC_RATES:
        if rate == "esd_isc":
            channels = [
                (n, 1, energies["S1"] - data["T1"].triplet(n), data["T1"].socme[(n, 1)], 1.0)
                for n in isc_channels(data["S1"], data["T1"], settings["esd_tn_window_ev"])
            ]
            combine = "sum"
        else:
            # T -> S: the three initial sublevels are equally populated.
            socme, sroot = (data["S1"].socme[(1, 1)], 1) if rate == "esd_risc" \
                else (data["S0"].socme[(1, 0)], 0)
            channels = [(1, sroot, energies["T1"] - energies[final], socme, 1 / math.sqrt(3))]
            combine = "mean"
        jobs, record, diagnostics = _isc(rate, channels, settings, files, temperature, sp_header)
        ht = any(job["ht"] for job in record)
        reasons = {job["triplet"]: job["ht_reason"] for job in record}
        ht_reason = next(iter(reasons.values())) if len(set(reasons.values())) == 1 \
            else "; ".join(f"T{n}: {reason}" for n, reason in sorted(reasons.items()))
    elif rate in ("esd_ic", "esd_fluor"):
        dele = (energies["S1"] - energies["S0"]) * EH_TO_CM
        ht, ht_reason = ht_decision(rate, 0.0, dele, settings)
        doht = {"DOHT": "TRUE"} if ht else {}
        if rate == "esd_ic":
            # NACMEs: full TDDFT with electron-translation factors, as the manual recommends.
            keyword, tddft, lineshape = "ESD(IC)", {"nacme": True, "etf": True, "tda": False}, LINESHAPE
        else:
            keyword, tddft, lineshape = "ESD(FLUOR)", {"tda": True}, {}
        jobs = [_tddft_job(
            sp_header, keyword,
            {"nroots": NROOTS[rate], "iroot": 1, **tddft},
            {**files, "DELE": _cm(dele), "USEJ": "TRUE", **doht, **lineshape, **temperature},
        )]
        record = [{"dele_cm": dele}]
        combine = "mean"
    else:  # esd_phosp: one job per T1 sublevel (SOC roots 1-3)
        dele = (energies["T1"] - energies["S0"]) * EH_TO_CM
        ht, ht_reason = ht_decision(rate, 0.0, dele, settings)
        doht = {"DOHT": "TRUE"} if ht else {}
        jobs = [
            _tddft_job(
                sp_header, "ESD(PHOSP)",
                {"nroots": NROOTS[rate], "iroot": k, "triplets": True, "dosoc": True, "tda": True},
                {**files, "DELE": _cm(dele), "USEJ": "TRUE", **doht, **temperature},
            )
            for k in (1, 2, 3)
        ]
        record = [{"sublevel": k, "dele_cm": dele} for k in (1, 2, 3)]
        combine = "mean"

    computed = {"combine": combine, "energies_eh": energies, "jobs": record,
                "ht": ht, "ht_reason": ht_reason}
    if diagnostics:
        computed["diagnostics"] = diagnostics
    return _join(jobs, charge), computed


def _isc(rate, channels, settings, files, temperature, sp_header):
    """ISC-type jobs per channel, then its FC diagnostics.

    Each channel is ``(triplet, sroot, gap (Eh), SOCME (cm-1), FC SOCME factor)``.
    FC jobs get DELE and SOCME; HT jobs compute SOC per triplet sublevel. Returns the
    jobs, a record per main job, and the diagnostics' ORCA job numbers and variants.

    ORCA's ``$new_job`` inherits every setting it does not repeat, so HT jobs come
    first (no inherited SOCME), FC jobs say DOHT FALSE, and TCUTFREQ comes last.
    """
    jobs: list[str] = []
    record: list[dict] = []
    fc_jobs = {}
    fc_main = []
    for n, sroot, gap, socme, factor in channels:
        dele = _cm(gap * EH_TO_CM)
        ht, reason = ht_decision(rate, socme, gap * EH_TO_CM, settings)
        common = {"triplet": n, "dele_cm": gap * EH_TO_CM, "ht": ht, "ht_reason": reason}
        fc_jobs[n] = _fc_job(files, dele, socme * factor, temperature)
        if not ht:
            fc_main.append({**common, "socme_cm": socme * factor})
            continue
        for ms in SUBLEVELS:
            jobs.append(_tddft_job(
                sp_header, "ESD(ISC)",
                {"nroots": max(sroot, n) + ROOTS_ABOVE, "sroot": sroot, "troot": n, "trootssl": ms,
                 "triplets": True, "dosoc": True, "tda": True},
                {**files, "DELE": dele, "USEJ": "TRUE", "DOHT": "TRUE", **LINESHAPE, **temperature},
            ))
            # ORCA computes each sublevel's SOC itself; the parser reads it back.
            record.append({**common, "sublevel": ms})
    main: dict[int, int] = {}
    for job in fc_main:
        jobs.append(fc_jobs[job["triplet"]](LINESHAPE))
        record.append(job)
        main[job["triplet"]] = len(jobs)
    diagnostics: list[dict] = []
    for variants in (("inlinew_100", "inlinew_25", "inlinew_250"), ("tcutfreq_300",)):
        for n, fc in fc_jobs.items():
            for variant in variants:
                if variant == "inlinew_100" and n in main:
                    job = main[n]  # the FC main job is its own baseline
                else:
                    jobs.append(fc(DIAGNOSTICS[variant]))
                    job = len(jobs)
                diagnostics.append({"job": job, "triplet": n, "variant": variant})
    jobs[len(record)] = DIAGNOSTICS_MARKER + jobs[len(record)]
    return jobs, record, diagnostics


def _fc_job(files, dele, socme_cm, temperature):
    """A builder of one channel's FC jobs, given their lineshape settings."""
    def job(lineshape):
        return FC_HEADER + esd_block(
            **files, DELE=dele, SOCME=f"0.0, {socme_cm / EH_TO_CM:.6e}", USEJ="TRUE", DOHT="FALSE",
            **lineshape, **temperature,
        )
    return job


def _cm(value: float) -> str:
    """*value* (cm-1) for a %esd block; a zero gap is 0.0, not -0.0."""
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def _tddft_job(sp_header: str, keyword: str, tddft: dict, esd: dict) -> str:
    """The singlepoint method plus an ESD keyword, a %tddft and a %esd block."""
    text = blocks.append_block(blocks.with_keyword(sp_header, keyword), blocks.tddft_block(**tddft))
    return blocks.append_block(text, esd_block(**esd))


def _join(jobs: list[str], charge: int) -> str:
    """Several ORCA jobs in one input; the input template appends the last geometry line."""
    geometry = f"*xyzfile {charge} 1 input.xyz\n"
    parts = [job.rstrip("\n") + "\n\n" + geometry for job in jobs[:-1]] + [jobs[-1]]
    return "\n$new_job\n".join(parts)
