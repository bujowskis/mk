import argparse
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib


# Note: this may be less efficient than running the evolution directly in Framsticks, so if performance is key, compare both options.


def genotype_within_constraint(genotype, dict_criteria_values, criterion_name, constraint_value):
	REPORT_CONSTRAINT_VIOLATIONS = False
	if constraint_value is not None:
		actual_value = dict_criteria_values[criterion_name]
		if actual_value > constraint_value:
			if REPORT_CONSTRAINT_VIOLATIONS:
				print('Genotype "%s" assigned low fitness because it violates constraint "%s": %s exceeds threshold %s' % (genotype, criterion_name, actual_value, constraint_value))
			return False
	return True


def frams_evaluate(frams_cli, individual):
	BAD_FITNESS = [-1] * len(OPTIMIZATION_CRITERIA)  # fitness of -1 is intended to discourage further propagation of this genotype via selection ("this genotype is very poor")
	genotype = individual[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	data = frams_cli.evaluate([genotype])
	# print("Evaluated '%s'" % genotype, 'evaluation is:', data)
	valid = True
	try:
		first_genotype_data = data[0]
		evaluation_data = first_genotype_data["evaluations"]
		default_evaluation_data = evaluation_data[""]
		fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
	except (KeyError, TypeError) as e:  # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
		valid = False
		print('Problem "%s" so could not evaluate genotype "%s", hence assigned it low fitness: %s' % (str(e), genotype, BAD_FITNESS))
	if valid:
		default_evaluation_data['numgenocharacters'] = len(genotype)  # for consistent constraint checking below
		valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numparts', parsed_args.max_numparts)
		valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numjoints', parsed_args.max_numjoints)
		valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numneurons', parsed_args.max_numneurons)
		valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numconnections', parsed_args.max_numconnections)
		valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numgenocharacters', parsed_args.max_numgenochars)
	if not valid:
		fitness = BAD_FITNESS
	return fitness


def frams_crossover(frams_cli, individual1, individual2):
	geno1 = individual1[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	geno2 = individual2[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	individual1[0] = frams_cli.crossOver(geno1, geno2)
	individual2[0] = frams_cli.crossOver(geno1, geno2)
	return individual1, individual2


def frams_mutate(frams_cli, individual):
	individual[0] = frams_cli.mutate([individual[0]])[0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	return individual,


def frams_getsimplest(frams_cli, genetic_format, initial_genotype):
	return initial_genotype if initial_genotype is not None else frams_cli.getSimplest(genetic_format)


def prepareToolbox(frams_cli, OPTIMIZATION_CRITERIA, tournament_size, genetic_format, initial_genotype):
	creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
	creator.create("Individual", list, fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"

	toolbox = base.Toolbox()
	toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_cli, genetic_format, initial_genotype)  # "Attribute generator"
	# (failed) struggle to have an individual which is a simple str, not a list of str
	# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
	# https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
	# https://github.com/DEAP/deap/issues/339
	# https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
	# https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", frams_evaluate, frams_cli)
	toolbox.register("mate", frams_crossover, frams_cli)
	toolbox.register("mutate", frams_mutate, frams_cli)
	if len(OPTIMIZATION_CRITERIA) <= 1:
		toolbox.register("select", tools.selTournament, tournsize=tournament_size)
	else:
		toolbox.register("select", tools.selNSGA2)
	return toolbox


def parseArguments():
	parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
	parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
	parser.add_argument('-sim', required=False, default="eval-allcriteria.sim", help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

	parser.add_argument('-genformat', required=False, help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.')
	parser.add_argument('-initialgenotype', required=False, help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.')

	parser.add_argument('-opt', required=True, help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria optimization, separate the names by the comma.')
	parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
	parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
	parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")
	parser.add_argument('-pmut', type=float, default=0.9, help="Probability of mutation, default: 0.9")
	parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")
	parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
	parser.add_argument('-hof_savefile', required=False, help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen).')

	parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of Parts. Default: no limit")
	parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of Joints. Default: no limit")
	parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of Neurons. Default: no limit")
	parser.add_argument('-max_numconnections', type=int, default=None, help="Maximum number of Neural connections. Default: no limit")
	parser.add_argument('-max_numgenochars', type=int, default=None, help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit")
	return parser.parse_args()


def ensureDir(string):
	if os.path.isdir(string):
		return string
	else:
		raise NotADirectoryError(string)


def save_genotypes(filename, OPTIMIZATION_CRITERIA, hof):
	from framsfiles import writer as framswriter
	with open(filename, "w") as outfile:
		for ind in hof:
			keyval = {}
			for i, k in enumerate(OPTIMIZATION_CRITERIA):  # construct a dictionary with criteria names and their values
				keyval[k] = ind.fitness.values[i]  # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
			# Note: prior to the release of Framsticks 5.0, saving e.g. numparts (i.e. P) without J,N,C breaks re-calcucation of P,J,N,C in Framsticks and they appear to be zero (nothing serious).
			outfile.write(framswriter.from_collection({"_classname": "org", "genotype": ind[0], **keyval}))
			outfile.write("\n")
	print("Saved '%s' (%d)" % (filename, len(hof)))


def main():
	global parsed_args, OPTIMIZATION_CRITERIA  # needed in frams_evaluate(), so made global to avoid passing as arguments

	# random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
	FramsticksLib.DETERMINISTIC = False  # must be set before FramsticksLib() constructor call
	parsed_args = parseArguments()
	print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))
	OPTIMIZATION_CRITERIA = parsed_args.opt.split(",")
	framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim.split(";"))
	toolbox = prepareToolbox(framsLib, OPTIMIZATION_CRITERIA, parsed_args.tournament, '1' if parsed_args.genformat is None else parsed_args.genformat, parsed_args.initialgenotype)
	pop = toolbox.population(n=parsed_args.popsize)
	hof = tools.HallOfFame(parsed_args.hof_size)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("stddev", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=parsed_args.pxov, mutpb=parsed_args.pmut, ngen=parsed_args.generations, stats=stats, halloffame=hof, verbose=True)
	print('Best individuals:')
	for ind in hof:
		print(ind.fitness, '\t-->\t', ind[0])
	if parsed_args.hof_savefile is not None:
		save_genotypes(parsed_args.hof_savefile, OPTIMIZATION_CRITERIA, hof)


if __name__ == "__main__":
	main()
