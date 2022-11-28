#!/bin/bash
#SBATCH --time=24:00:00
GEN=$1 #GEN
OPT=$2 #OPT
FRAMS_PATH="path"
SIM=$4 #SIM
FIT=$5 #fit
DISSIM=$6 #DISSIM
ARCHIVE=$7
GEERATIONS=2000
NORM=$8

# srun -p idss-student python3 -m evolalg.run_niching -path $FRAMS_PATH -sim $SIM -opt $OPT -genformat $GEN -dissim $DISSIM -fit $FIT -archive $ARCHIVE -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 100 -generations $GEERATIONS -normalize $NORM -hof_savefile states/${GEN}_${OPT}_${FIT}_${DISSIM}_${ARCHIVE}_${SLURM_ARRAY_TASK_ID}.gen 
python3 -m evolalg.run_niching -path $FRAMS_PATH -sim $SIM -opt $OPT -genformat $GEN -dissim $DISSIM -fit $FIT -archive $ARCHIVE -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 100 -generations $GEERATIONS -normalize $NORM -hof_savefile states/${GEN}_${OPT}_${FIT}_${DISSIM}_${ARCHIVE}_${SLURM_ARRAY_TASK_ID}.gen
