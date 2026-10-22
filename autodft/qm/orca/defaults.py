"""Default ORCA input headers for common calculation types.

Also holds the QZVPD and GFN2-xTB alternates, the ``LEGACY_*`` full
headers of before Advanced settings, and the ``SEED_HEADERS`` rows a fresh
database is seeded with.
"""

# Method only: grid, convergence, RI, solvation, cores and memory are Advanced settings.
# Conformer search at g-xTB through ORCA's GOAT driver: the bare "!GOAT XTB"
# tag plus ``%xtb XTBInputString "--gxtb" end``.
DEFAULT_HEADER_CONFSEARCH = (
    "!GOAT XTB\n"
    "%xtb\n"
    '  XTBInputString "--gxtb"\n'
    "end\n"
    "%GOAT\n"
    "MAXEN 10.0\n"
    "ENDIFF 0.2\n"
    "RMSD 0.15\n"
    "FREEZECISTRANS TRUE\n"
    "CONFDEGEN AUTO\n"
    "END\n"
)

DEFAULT_HEADER_OPTIMIZATION = "!wB97X-D3 def2-SVP\n"

DEFAULT_HEADER_SINGLEPOINT = "!wB97X-D3 def2-TZVP\n"


# The full headers of before Advanced settings; running projects use them.
LEGACY_HEADER_CONFSEARCH = (
    "!GOAT XTB\n"
    "%xtb\n"
    '  XTBInputString "--gxtb"\n'
    "end\n"
    "%maxcore 500\n"
    "%pal nprocs 32 end\n"
    "%GOAT\n"
    "MAXEN 10.0\n"
    "ENDIFF 0.2\n"
    "RMSD 0.15\n"
    "FREEZECISTRANS TRUE\n"
    "CONFDEGEN AUTO\n"
    "END\n"
)

LEGACY_HEADER_OPTIMIZATION = (
    "!wB97X-D3 def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF CPCM(MeCN) Freq\n"
    "%maxcore 2000\n"
    "%pal nprocs 12 end\n"
)

LEGACY_HEADER_SINGLEPOINT = (
    # Singlepoint headers must NEVER include "Opt" or "Freq" — those would
    # turn the supposedly cheap singlepoint into a re-optimization or a
    # full Hessian calculation on the produced geometry.
    "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n"
    "%maxcore 1500\n"
    "%pal nprocs 8 end\n"
)


# The larger basis, kept as the alternate singlepoint row.
QZVPD_HEADER_SINGLEPOINT = "!wB97X-D3 def2-QZVPD\n"


# The same conformer search at GFN2-xTB, kept as the alternate.
GFN2_HEADER_CONFSEARCH = (
    "!GOAT XTB2\n"
    "%GOAT\n"
    "MAXEN 10.0\n"
    "ENDIFF 0.2\n"
    "RMSD 0.15\n"
    "FREEZECISTRANS TRUE\n"
    "CONFDEGEN AUTO\n"
    "END\n"
)

LEGACY_GFN2_HEADER_CONFSEARCH = (
    "!GOAT XTB2\n"
    "%maxcore 500\n"
    "%pal nprocs 32 end\n"
    "%GOAT\n"
    "MAXEN 10.0\n"
    "ENDIFF 0.2\n"
    "RMSD 0.15\n"
    "FREEZECISTRANS TRUE\n"
    "CONFDEGEN AUTO\n"
    "END\n"
)


# Headers seeded into a fresh ``computation_headers`` table; init_db() adds
# any missing ones to an existing table. Each entry is shown in
# the dashboard's Headers manager and the kind-filtered submission
# dropdowns.
SEED_HEADERS = [
    {
        "kind": "confsearch",
        "description": "GOAT GFN2-xTB conformer ensemble",
        "header_text": GFN2_HEADER_CONFSEARCH,
    },
    {
        "kind": "confsearch",
        "description": "GOAT g-xTB conformer ensemble",
        "header_text": DEFAULT_HEADER_CONFSEARCH,
    },
    {
        "kind": "singlepoint",
        "description": "wB97X-D3 / def2-QZVPD",
        "header_text": QZVPD_HEADER_SINGLEPOINT,
    },
    {
        "kind": "optimization",
        "description": "wB97X-D3 / def2-SVP",
        "header_text": DEFAULT_HEADER_OPTIMIZATION,
    },
    {
        "kind": "singlepoint",
        "description": "wB97X-D3 / def2-TZVP",
        "header_text": DEFAULT_HEADER_SINGLEPOINT,
    },
]
