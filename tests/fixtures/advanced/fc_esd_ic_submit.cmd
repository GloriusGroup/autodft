#!/bin/bash
#SBATCH --job-name=autodft_N_esd_ic
#SBATCH --output=/ROOT/rates/esd_ic/job_N/slurm-%j.out
#SBATCH --error=/ROOT/rates/esd_ic/job_N/slurm-%j.err
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12400
#SBATCH --time=10-00:00:00
#SBATCH --partition=CPU
#SBATCH --nice=1000


# Enable nullglob so unmatched globs in the cp lines below don't expand to themselves
shopt -s nullglob

WORK_DIR="/ROOT/rates/esd_ic/job_N"
TMP_DIR="/tmp/autodft_N_esd_ic.${SLURM_JOB_ID:-$$}"

export NBOEXE=/path/to/nbo7.i8.exe


mkdir -p "$TMP_DIR"
cp "$WORK_DIR"/input.inp "$TMP_DIR"/
cp "$WORK_DIR"/gs.hess "$TMP_DIR"/
cp "$WORK_DIR"/es.hess "$TMP_DIR"/
if [ -f "$WORK_DIR"/input.xyz ]; then
    cp "$WORK_DIR"/input.xyz "$TMP_DIR"/
    cp "$WORK_DIR"/input.xyz "$TMP_DIR"/input_original.xyz
fi

cd "$TMP_DIR"


# Run ORCA — full path or 'orca' from the [orca].path config field.
orca input.inp > output.out
ORCA_EXIT=$?


# Copy results back to the shared job directory
cp *.out *.txt *.xyz "$WORK_DIR"/ 2>/dev/null || true
cp *.FS.hess *.GS.hess *.ES.hess *.spectrum* *.corrfunc "$WORK_DIR"/ 2>/dev/null || true
cp input.finalensemble.xyz              "$WORK_DIR"/ 2>/dev/null || true

cd /
rm -rf "$TMP_DIR"


exit $ORCA_EXIT
