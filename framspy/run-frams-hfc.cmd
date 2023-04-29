set DIR_WITH_FRAMS_LIBRARY="C:\Users\sofya\Desktop\MK\Framsticks50rc26"

python -m evolalg.run_CSvsHFC_frams_hfc -path %DIR_WITH_FRAMS_LIBRARY% -sim eval-allcriteria-mini.sim -opt vertpos -genformat 1 -max_numparts 15 -initialgenotype /*9*/BLU -max_numneurons 15 -max_numjoints 5 -max_numconnections 5  -max_numgenochars 100 -popsize 20 -generations 15 -mutsize 0.01 -runnum 1