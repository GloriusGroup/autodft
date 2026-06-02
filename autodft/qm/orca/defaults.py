"""Default ORCA input headers for common calculation types.

Copied verbatim from the legacy
``/mnt/share/dft_calculations/scripts/job_submission/submit_to_db_zmy.py``
submission script so this package reproduces what's been running in
production on this cluster.
"""

DEFAULT_HEADER_CONFSEARCH = (
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

DEFAULT_HEADER_OPTIMIZATION = (
    "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightOpt TightSCF Freq\n"
    "%maxcore 1000\n"
    "%pal nprocs 8 end\n"
)

DEFAULT_HEADER_SINGLEPOINT = (
    # Singlepoint headers must NEVER include "Opt" or "Freq" — those would
    # turn the supposedly cheap singlepoint into a re-optimization or a
    # full Hessian calculation on the produced geometry.
    "!wB97X-D3 def2-QZVPD def2/J RIJCOSX DEFGRID3 TightSCF KeepDens\n"
    "%maxcore 1500\n"
    "%pal nprocs 2 end\n"
)


# Secondary B3LYP defaults — cheaper alternative for both optimisation
# and singlepoint stages.
B3LYP_HEADER_OPTIMIZATION = (
    "!B3LYP def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF Freq\n"
    "%maxcore 500\n"
    "%pal nprocs 8 end\n"
)

B3LYP_HEADER_SINGLEPOINT = (
    "!B3LYP def2-TZVP def2/J RIJCOSX DEFGRID3 TightSCF\n"
    "%maxcore 500\n"
    "%pal nprocs 2 end\n"
)


# Additional conformer search variant using g-xTB through ORCA's GOAT
# driver. Useful for systems where GFN2-xTB struggles. Activated via
# ``%xtb XTBInputString "--gxtb" end`` and the bare "!GOAT XTB" tag.
GXTB_HEADER_CONFSEARCH = (
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


# Headers seeded into the database the first time init_db() is called
# against a fresh ``computation_headers`` table. Each entry is shown in
# the dashboard's Headers manager and the kind-filtered submission
# dropdowns.
SEED_HEADERS = [
    {
        "kind": "confsearch",
        "description": "GOAT GFN2-xTB conformer ensemble",
        "header_text": DEFAULT_HEADER_CONFSEARCH,
    },
    {
        "kind": "confsearch",
        "description": "GOAT g-xTB conformer ensemble",
        "header_text": GXTB_HEADER_CONFSEARCH,
    },
    {
        "kind": "optimization",
        "description": "wB97X-D3 / def2-TZVP TightOpt + Freq",
        "header_text": DEFAULT_HEADER_OPTIMIZATION,
    },
    {
        "kind": "optimization",
        "description": "B3LYP / def2-SVP Opt + Freq",
        "header_text": B3LYP_HEADER_OPTIMIZATION,
    },
    {
        "kind": "singlepoint",
        "description": "wB97X-D3 / def2-QZVPD KeepDens",
        "header_text": DEFAULT_HEADER_SINGLEPOINT,
    },
    {
        "kind": "singlepoint",
        "description": "B3LYP / def2-TZVP",
        "header_text": B3LYP_HEADER_SINGLEPOINT,
    },
]
