python3 -m evolalg.run_niching -path C:\Users\ADMIN\Downloads\Framsticks50rc23 -opt vertpos -genformat $GEN -dissim $DISSIM -fit $FIT -archive $ARCHIVE -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 100 -generations 2000 -normalize $NORM -hof_savefile states/${GEN}_${OPT}_${FIT}_${DISSIM}_${ARCHIVE}_${SLURM_ARRAY_TASK_ID}.gen