set DIR_WITH_FRAMS_LIBRARY="C:\Users\sofya\Desktop\MK\Framsticks50rc23"

python -m evolalg.run_CSvsHFC_frams -path %DIR_WITH_FRAMS_LIBRARY% -islands=10 -generations_migration=5 -sim eval-allcriteria-mini.sim -opt vertpos -genformat 1 -max_numparts 15 -max_numneurons 15 -max_numjoints 5 -max_numconnections 5  -max_numgenochars 100 -popsize 20 -generations 15 -mutsize 0.01 -runnum 1
