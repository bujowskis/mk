set DIR_WITH_FRAMS_LIBRARY=%FRAMSTICKS_DIR%
set HOF_FOLDER = HoF/numerical_complex

rem -save_results ???

for %%i in (1,2,3,4,5) do (
    python3 -m evolalg.run_numerical_complex -dimensions %%i -benchmark_function "Ackley" -hof_savefile HoF/numerical_complex/HoF-numerical_complex-Ackley-%%i.gen -generations 10000 -popsize 50
)
for %%i in (1,2,3,4,5) do (
    python3 -m evolalg.run_numerical_complex -dimensions %%i -benchmark_function "EggHolder" -hof_savefile HoF/numerical_complex/HoF-numerical_complex-EggHolder-%%i.gen -generations 10000 -popsize 50
)

python3 -m evolalg.run_numerical_complex -benchmark_function "Schaffer2" -hof_savefile HoF/numerical_complex/HoF-numerical_complex-Schaffer.gen -generations 10000 -popsize 50

rem python3 -m evolalg.run_frams_islands  -path $FRAMS_PATH -n_pop=$N_POP -m_interv=$M_INTERV -sim $SIM -opt $OPT -genformat $GEN -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 100 -generations $GENERATIONS  -hof_savefile states/${GEN}_${OPT}_${N_POP}_${SLURM_ARRAY_TASK_ID}.gen
