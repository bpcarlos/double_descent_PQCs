#!/bin/bash
####################################################################
# SLURM options:
#
# Name of the job
#SBATCH -J L11
#
# Number of tasks 
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
#
# Number of processors
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#
# File for std and error output
#SBATCH -e err.job.%j
#SBATCH -o log.job.%j
#
# e-mail address
#SBATCH --mail-user=c.bravo.prieto@fu-berlin.de
#SBATCH --mail-type=END,FAIL
#
###################################################################

set -eu

source ~/ddpqc/bin/activate

python3 main_code.py

exit 0


