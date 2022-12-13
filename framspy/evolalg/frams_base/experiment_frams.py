from ..base.experiment_abc import ExperimentABC
from ..constants import BAD_FITNESS
from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..utils import ensureDir


class ExperimentFrams(ExperimentABC):
    def __init__(self, hof_size, popsize, frams_lib, optimization_criteria, genformat, save_only_best=True, constraints={}) -> None:
        ExperimentABC.__init__(self, hof_size=hof_size, popsize=popsize, save_only_best=save_only_best)
        self.optimization_criteria = optimization_criteria
        self.frams_lib = frams_lib
        self.constraints = constraints
        self.genformat = genformat

    def frams_getsimplest(self, genetic_format, initial_genotype):
        return initial_genotype if initial_genotype is not None else self.frams_lib.getSimplest(genetic_format)

    def genotype_within_constraint(self, genotype, dict_criteria_values, criterion_name, constraint_value):
        REPORT_CONSTRAINT_VIOLATIONS = False
        if constraint_value is not None:
            actual_value = dict_criteria_values[criterion_name]
            if actual_value > constraint_value:
                if REPORT_CONSTRAINT_VIOLATIONS:
                    print('Genotype "%s" assigned low fitness because it violates constraint "%s": %s exceeds threshold %s' % (
                        genotype, criterion_name, actual_value, constraint_value))
                return False
        return True

    def check_valid_constraints(self, genotype, default_evaluation_data):
        valid = True
        valid &= self.genotype_within_constraint(
            genotype, default_evaluation_data, 'numparts', self.constraints.get('max_numparts'))
        valid &= self.genotype_within_constraint(
            genotype, default_evaluation_data, 'numjoints', self.constraints.get('max_numjoints'))
        valid &= self.genotype_within_constraint(
            genotype, default_evaluation_data, 'numneurons', self.constraints.get('max_numneurons'))
        valid &= self.genotype_within_constraint(
            genotype, default_evaluation_data, 'numconnections', self.constraints.get('max_numconnections'))
        valid &= self.genotype_within_constraint(
            genotype, default_evaluation_data, 'numgenocharacters', self.constraints.get('max_numgenochars'))
        return valid

    def evaluate(self, genotype):
        data = self.frams_lib.evaluate([genotype])
        # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
        valid = True
        try:
            first_genotype_data = data[0]
            evaluation_data = first_genotype_data["evaluations"]
            default_evaluation_data = evaluation_data[""]
            fitness = [default_evaluation_data[crit] for crit in self.optimization_criteria][0]
        # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
        except (KeyError, TypeError) as e:
            valid = False
            print('Problem "%s" so could not evaluate genotype "%s", hence assigned it fitness: %s' % (
                str(e), genotype, BAD_FITNESS))
        if valid:
            default_evaluation_data['numgenocharacters'] = len(genotype)  # for consistent constraint checking below
            valid = self.check_valid_constraints(genotype, default_evaluation_data) 
        if not valid:
            fitness = BAD_FITNESS
        return fitness
        

    def mutate(self, gen1):
        return self.frams_lib.mutate([gen1])[0]

    def cross_over(self, gen1, gen2):
        return self.frams_lib.crossOver(gen1, gen2)

    def initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.time_elapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.set_and_evaluate(self.frams_getsimplest(
            '1' if self.genformat is None else self.genformat, initialgenotype), self.evaluate)
        self.hof.add(initial_individual)
        self.stats.append(
            initial_individual.rawfitness if self.save_only_best else initial_individual)
        self.population_structures = PopulationStructures(
            initial_individual=initial_individual, popsize=self.popsize)

    def save_genotypes(self, filename):
        from framsfiles import writer as framswriter
        with open(filename, "w") as outfile:
            for ind in self.hof:
                keyval = {}
                # construct a dictionary with criteria names and their values
                for i, k in enumerate(self.optimization_criteria):
                    # .values[i]  # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
                    keyval[k] = ind.rawfitness
                # Note: prior to the release of Framsticks 5.0, saving e.g. numparts (i.e. P) without J,N,C breaks re-calcucation of P,J,N,C in Framsticks and they appear to be zero (nothing serious).
                outfile.write(framswriter.from_collection(
                    {"_classname": "org", "genotype": ind.genotype, **keyval}))
                outfile.write("\n")
        print("Saved '%s' (%d)" % (filename, len(self.hof)))

    @staticmethod
    def get_args_for_parser():
        parser = ExperimentABC.get_args_for_parser()
        parser.add_argument('-path',type= ensureDir, required= True,
                        help= 'Path to Framsticks CLI without trailing slash.')
        parser.add_argument('-lib',type= str, required= False,
                        help= 'Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
        parser.add_argument('-sim',type= str, required= False, default= "eval-allcriteria.sim",
                        help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

        parser.add_argument('-genformat',type= str, required= False,
                            help= 'Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.')
        parser.add_argument('-initialgenotype',type= str, required= False,
                                    help= 'The genotype used to seed the initial population. If given, the -genformat argument is ignored.')
        parser.add_argument('-opt',required=True, help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef).')

        parser.add_argument('-max_numparts',type= int, default= None,
                                help="Maximum number of Parts. Default: no limit")
        parser.add_argument('-max_numjoints',type= int, default= None,
                                help="Maximum number of Joints. Default: no limit")
        parser.add_argument('-max_numneurons',type= int, default= None,
                                help="Maximum number of Neurons. Default: no limit")
        parser.add_argument('-max_numconnections',type= int, default= None,
                                    help="Maximum number of Neural connections. Default: no limit")
        parser.add_argument('-max_numgenochars',type= int, default= None,
                                    help="Maximum number of characters in genotype (including the format prefix, if any}. Default: no limit")
        return parser