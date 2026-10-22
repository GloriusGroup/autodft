#!/bin/bash
#SBATCH --job-name=autodft_N_esd_isc_t1s0
#SBATCH --output=/ROOT/rates/esd_isc_t1s0_part3/job_N/slurm-%j.out
#SBATCH --error=/ROOT/rates/esd_isc_t1s0_part3/job_N/slurm-%j.err
#SBATCH --ntasks-per-node=12
#SBATCH --mem=24600
#SBATCH --time=10-00:00:00
#SBATCH --partition=CPU
#SBATCH --nice=1000


# Enable nullglob so unmatched globs in the cp lines below don't expand to themselves
shopt -s nullglob

WORK_DIR="/ROOT/rates/esd_isc_t1s0_part3/job_N"
TMP_DIR="/tmp/autodft_N_esd_isc_t1s0.${SLURM_JOB_ID:-$$}"

export NBOEXE=/path/to/nbo7.i8.exe


mkdir -p "$TMP_DIR"
cp "$WORK_DIR"/input.inp "$TMP_DIR"/
cp "$WORK_DIR"/initial.hess "$TMP_DIR"/
cp "$WORK_DIR"/final.hess "$TMP_DIR"/
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
