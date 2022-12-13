rem To learn about all available options of each .py algorithm below, add "-h" to its parameters.
rem Use the source code of the examples as a starting point for your customizations.
rem Example usage:

set DIR_WITH_FRAMS_LIBRARY=%FRAMSTICKS_DIR%




rem evolution with niching
python -m evolalg.run_frams_niching  -path %DIR_WITH_FRAMS_LIBRARY%  -sim eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim -opt velocity -genformat 0 -dissim 1 -fit knn_niching -archive 50 -max_numparts 15 -max_numneurons 15 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 10000 -popsize 50 -generations 10 -normalize none -hof_savefile HoF-niching.gen 

rem a generic island model example
python -m evolalg.run_frams_islands  -path %DIR_WITH_FRAMS_LIBRARY%  -islands=10 -generations_migration=5 -sim eval-allcriteria.sim;only-body.sim;deterministic.sim;sample-period-2.sim -opt vertpos -genformat 1 -max_numparts 15 -max_numneurons 15 -max_numjoints 5 -max_numconnections 5  -max_numgenochars 100 -popsize 20 -generations 15  -hof_savefile HoF-islands.gen

rem a combination of various parameters for thorough testing, see source of 'test_diferent_settings.py'
rem python -m evolalg.tests.test_diferent_settings  -path %DIR_WITH_FRAMS_LIBRARY%  -sim eval-allcriteria.sim



rem numerical examples (unrelated to Framsticks)
python -m evolalg.run_numerical_example
python -m evolalg.run_numerical_islands_example
