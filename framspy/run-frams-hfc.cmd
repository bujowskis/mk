set DIR_WITH_FRAMS_LIBRARY="C:\Users\sofya\Desktop\MK\Framsticks50rc28"

python -m evolalg.run_CSvsHFC_frams_hfc -path %DIR_WITH_FRAMS_LIBRARY% -opt vertpos -genformat 1 -max_numparts 15 -max_numneurons 20 -max_numjoints 30 -max_numconnections 30  -max_numgenochars 1000 -popsize 20 -generations 50 -runnum 1 -tsize 10
