#!/bin/bash
#SBATCH --time=24:00:00
GEN=$1 #GEN
OPT=$2 #OPT
FRAMS_PATH=""
SIM=$4 #SIM
GENERATIONS=100
N_POP=5
M_INTERV=10

python3 -m evolalg.run_frams_islands  -path $FRAMS_PATH -n_pop=$N_POP -m_interv=$M_INTERV -sim $SIM -opt $OPT -genformat $GEN -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 100 -generations $GENERATIONS  -hof_savefile states/${GEN}_${OPT}_${N_POP}_${SLURM_ARRAY_TASK_ID}.gen 
