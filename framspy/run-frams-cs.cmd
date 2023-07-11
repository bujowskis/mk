set DIR_WITH_FRAMS_LIBRARY="C:\Users\sofya\Desktop\MK\Framsticks50rc28"

python -m evolalg.run_CSvsHFC_frams -path %DIR_WITH_FRAMS_LIBRARY% -sim "eval-allcriteria-mini.sim;deterministic.sim;sample-period-2.sim;only-body.sim" -opt vertpos -genformat 1   -max_numparts 15 -max_numjoints 30 -max_numneurons 20 -max_numconnections 30 -max_numgenochars 1000   -popsize 20 -generations 15 -mutsize 0.01 -runnum 1
