from ..base.experiment_abc import ExperimentABC, STATS_SAVE_ONLY_BEST_FITNESS
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..structures.hall_of_fame import HallOfFame

BAD_FITNESS = None

class ExperimentFrams(ExperimentABC):
    def __init__(self, frams_lib, optimization_criteria, hof_size, popsize, genformat, constraints={}) -> None:
        self.optimization_criteria = optimization_criteria 
        self.frams_lib = frams_lib
        self.constraints = constraints
        self.hof = HallOfFame(hof_size)
        self.popsize = popsize
        self.genformat = genformat

        
    def frams_getsimplest(self, genetic_format, initial_genotype):
        return initial_genotype if initial_genotype is not None else self.frams_lib.getSimplest(genetic_format)

    def genotype_within_constraint(self, genotype, dict_criteria_values, criterion_name, constraint_value):
        REPORT_CONSTRAINT_VIOLATIONS = False
        if constraint_value is not None:
            actual_value = dict_criteria_values[criterion_name]
            if actual_value > constraint_value:
                if REPORT_CONSTRAINT_VIOLATIONS:
                    print('Genotype "%s" assigned low fitness because it violates constraint "%s": %s exceeds threshold %s' % (genotype, criterion_name, actual_value, constraint_value))
                return False
        return True

    def check_valid_constraints(self,genotype,default_evaluation_data):
        valid = True
        valid &= self.genotype_within_constraint(genotype, default_evaluation_data, 'numparts', self.constraints.get('max_numparts'))
        valid &= self.genotype_within_constraint(genotype, default_evaluation_data, 'numjoints', self.constraints.get('max_numjoints'))
        valid &= self.genotype_within_constraint(genotype, default_evaluation_data, 'numneurons', self.constraints.get('max_numneurons'))
        valid &= self.genotype_within_constraint(genotype, default_evaluation_data, 'numconnections', self.constraints.get('max_numconnections'))
        valid &= self.genotype_within_constraint(genotype, default_evaluation_data, 'numgenocharacters', self.constraints.get('max_numgenochars'))
        return valid

    def evaluate(self, genotype):
        data = self.frams_lib.evaluate([genotype])
        # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
        valid = True
        try:
            first_genotype_data = data[0]
            evaluation_data = first_genotype_data["evaluations"]
            default_evaluation_data = evaluation_data[""]
            fitness = [default_evaluation_data[crit] for crit in self.optimization_criteria]
        except (KeyError, TypeError) as e:  # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
            valid = False
            print('Problem "%s" so could not evaluate genotype "%s", hence assigned it fitness: %s' % (str(e), genotype, BAD_FITNESS))
        if valid: #TODO Refactor/Change to dict
            default_evaluation_data['numgenocharacters'] = len(genotype)  # for consistent constraint checking below
            valid = self.check_valid_constraints(genotype,default_evaluation_data)
        if not valid:
            fitness = BAD_FITNESS
        return fitness

    def get_state(self):
        return [self.timeelapsed, self.current_generation,self.current_population,self.hof,self.stats]
    
    def set_state(self,state):
        self.timeelapsed, self.current_generation,self.current_population,hof_,self.stats = state
        for h in sorted(hof_, key=lambda x: x.rawfitness):  # sorting: ensure that we add from worst to best so all individuals are added to HOF
            self.hof.add(h)

    def mutate(self, gen1):
        return self.frams_lib.mutate([gen1])[0]

    def cross_over(self, gen1, gen2):
        return self.frams_lib.crossOver(gen1, gen2)

    def _initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.setAndEvaluate(self.frams_getsimplest('1' if self.genformat is None else self.genformat, initialgenotype), self.evaluate)
        self.hof.add(initial_individual)
        self.stats.append(initial_individual.rawfitness if STATS_SAVE_ONLY_BEST_FITNESS else initial_individual)
        self.current_population = PopulationStructures(initial_individual=initial_individual, archive_size=0, popsize=self.popsize)
