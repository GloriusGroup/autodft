"""Default ORCA input headers for common calculation types.

Also holds the larger-basis and g-xTB alternates, and the ``SEED_HEADERS``
rows a fresh database is seeded with.
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
    "!wB97X-D3 def2-SVP def2/J RIJCOSX DEFGRID3 Opt TightSCF CPCM(MeCN) Freq\n"
    "%maxcore 1000\n"
    "%pal nprocs 8 end\n"
)

DEFAULT_HEADER_SINGLEPOINT = (
    # Singlepoint headers must NEVER include "Opt" or "Freq" — those would
    # turn the supposedly cheap singlepoint into a re-optimization or a
    # full Hessian calculation on the produced geometry.
    "!wB97X-D3 def2-TZVPD def2/J RIJCOSX DEFGRID3 TightSCF CPCM(MeCN) KeepDens\n"
    "%maxcore 1500\n"
    "%pal nprocs 2 end\n"
)


# Optional pair for the larger basis sets, kept as the second choice in
# each slot's seeded rows.
TZVP_MECN_HEADER_OPTIMIZATION = (
    "!wB97X-D3 def2-TZVP def2/J RIJCOSX DEFGRID3 TightOpt TightSCF CPCM(MeCN) Freq\n"
    "%maxcore 1000\n"
    "%pal nprocs 8 end\n"
)

QZVPD_MECN_HEADER_SINGLEPOINT = (
    "!wB97X-D3 def2-QZVPD def2/J RIJCOSX DEFGRID3 TightSCF KeepDens CPCM(MeCN)\n"
    "%maxcore 1500\n"
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
        "description": "wB97X-D3 / def2-TZVP TightOpt + Freq + CPCM(MeCN)",
        "header_text": TZVP_MECN_HEADER_OPTIMIZATION,
    },
    {
        "kind": "singlepoint",
        "description": "wB97X-D3 / def2-QZVPD KeepDens + CPCM(MeCN)",
        "header_text": QZVPD_MECN_HEADER_SINGLEPOINT,
    },
    {
        "kind": "optimization",
        "description": "wB97X-D3 / def2-SVP Opt + Freq + CPCM(MeCN)",
        "header_text": DEFAULT_HEADER_OPTIMIZATION,
    },
    {
        "kind": "singlepoint",
        "description": "wB97X-D3 / def2-TZVPD KeepDens + CPCM(MeCN)",
        "header_text": DEFAULT_HEADER_SINGLEPOINT,
    },
]
