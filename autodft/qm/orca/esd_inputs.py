"""ORCA ESD rate inputs, built from the SOC singlepoints and Hessians of S0*, S1 and T1.

Energies are TDDFT-consistent: E(S0) is the ground state at S0*, E(S1) the
S1 root at the S1 geometry, E(T_n) the T_n root at the T1 geometry (T_n for
n >= 2 borrows the T1 geometry and Hessian: the shifted-T1 proxy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    ht = settings[categories.ESD_HT]
    initial, final, hessians = RATES[rate]
    files = {_HESSIAN_KEYS[name]: f'"{name}"' for name in hessians}
    temperature = {"TEMP": f"{settings['esd_temperature_k']:g}"}
    doht = {"DOHT": "TRUE"} if ht else {}
    energies = {state: energy(state, data[state]) for state in (initial, final)}

    if rate == "esd_isc":
        channels = [
            (n, energies["S1"] - data["T1"].triplet(n), data["T1"].socme[(n, 1)])
            for n in isc_channels(data["S1"], data["T1"], settings["esd_tn_window_ev"])
        ]
        jobs, record = _isc(channels, 1, ht, files, doht, temperature, sp_header)
        combine = "sum"
    elif rate in ("esd_risc", "esd_isc_t1s0"):
        # T -> S: the three initial sublevels are equally populated.
        if rate == "esd_risc":
            socme, sroot = data["S1"].socme[(1, 1)], 1
        else:
            socme, sroot = data["S0"].socme[(1, 0)], 0
        gap = energies["T1"] - energies[final]
        jobs, record = _isc([(1, gap, socme / math.sqrt(3))], sroot, ht, files, doht,
                            temperature, sp_header)
        combine = "mean"
    elif rate in ("esd_ic", "esd_fluor"):
        dele = (energies["S1"] - energies["S0"]) * EH_TO_CM
        if rate == "esd_ic":
            keyword, tddft, doht = "ESD(IC)", {"nacme": True, "etf": True}, {}
        else:
            keyword, tddft = "ESD(FLUOR)", {}
        jobs = [_tddft_job(
            sp_header, keyword,
            {"nroots": blocks.ESD_NROOTS, "iroot": 1, **tddft, "tda": False},
            {**files, "DELE": f"{dele:.1f}", "USEJ": "TRUE", **doht, **temperature},
        )]
        record = [{"dele_cm": dele}]
        combine = "mean"
    else:  # esd_phosp: one job per T1 sublevel (SOC roots 1-3)
        dele = (energies["T1"] - energies["S0"]) * EH_TO_CM
        jobs = [
            _tddft_job(
                sp_header, "ESD(PHOSP)",
                {"nroots": blocks.ESD_NROOTS, "iroot": k, "triplets": True, "dosoc": True,
                 "tda": False},
                {**files, "DELE": f"{dele:.1f}", "USEJ": "TRUE", **doht, **temperature},
            )
            for k in (1, 2, 3)
        ]
        record = [{"sublevel": k, "dele_cm": dele} for k in (1, 2, 3)]
        combine = "mean"

    computed = {"combine": combine, "energies_eh": energies, "jobs": record}
    return _join(jobs, charge), computed


def _isc(channels, sroot, ht, files, doht, temperature, sp_header) -> tuple[list[str], list[dict]]:
    """ISC-type jobs: FC gets DELE and SOCME; HT computes SOC per triplet sublevel."""
    jobs: list[str] = []
    record: list[dict] = []
    for n, gap, socme in channels:
        dele = f"{gap * EH_TO_CM:.1f}"
        if not ht:
            jobs.append(FC_HEADER + esd_block(
                **files, DELE=dele, SOCME=f"0.0, {socme / EH_TO_CM:.6e}", USEJ="TRUE",
                **temperature,
            ))
            record.append({"triplet": n, "dele_cm": gap * EH_TO_CM, "socme_cm": socme})
            continue
        for ms in SUBLEVELS:
            jobs.append(_tddft_job(
                sp_header, "ESD(ISC)",
                {"nroots": blocks.ESD_NROOTS, "sroot": sroot, "troot": n, "trootssl": ms,
                 "triplets": True, "dosoc": True, "tda": False},
                {**files, "DELE": dele, "USEJ": "TRUE", **doht, **temperature},
            ))
            record.append({"triplet": n, "sublevel": ms, "dele_cm": gap * EH_TO_CM,
                           "socme_cm": socme})
    return jobs, record


def _tddft_job(sp_header: str, keyword: str, tddft: dict, esd: dict) -> str:
    """The singlepoint method plus an ESD keyword, a %tddft and a %esd block."""
    text = blocks.append_block(blocks.with_keyword(sp_header, keyword), blocks.tddft_block(**tddft))
    return blocks.append_block(text, esd_block(**esd))


def _join(jobs: list[str], charge: int) -> str:
    """Several ORCA jobs in one input; the input template appends the last geometry line."""
    geometry = f"*xyzfile {charge} 1 input.xyz\n"
    parts = [job.rstrip("\n") + "\n\n" + geometry for job in jobs[:-1]] + [jobs[-1]]
    return "\n$new_job\n".join(parts)
