from typing import List  # to be able to specify a type hint of list(something)
import json
import sys, os
import argparse
import numpy as np
import frams


class FramsticksLib:
	"""Communicates directly with Framsticks library (.dll or .so).
	You can perform basic operations like mutation, crossover, and evaluation of genotypes.
	This way you can perform evolution controlled by python as well as access and manipulate genotypes.
	You can even design and use in evolution your own genetic representation implemented entirely in python,
	or access and control the simulation and simulated creatures step by step.

	Should you want to modify or extend this class, first see and test the examples in frams-test.py.

	You need to provide one or two parameters when you run this class: the path to Framsticks where .dll/.so resides
	and, optionally, the name of the Framsticks dll/so (if it is non-standard). See::
		FramsticksLib.py -h"""

	PRINT_FRAMSTICKS_OUTPUT: bool = False  # set to True for debugging
	DETERMINISTIC: bool = False  # set to True to have the same results in each run

	GENOTYPE_INVALID = "/*invalid*/"  # this is how genotype invalidity is represented in Framsticks
	EVALUATION_SETTINGS_FILE = [  # all files MUST be compatible with the standard-eval expdef. The order they are loaded in is important!
		"eval-allcriteria.sim",  # a good trade-off in performance sampling period ("perfperiod") for vertpos and velocity
		# "deterministic.sim",  # turns off random noise (added for robustness) so that each evaluation yields identical performance values (causes "overfitting")
		# "sample-period-2.sim", # short performance sampling period so performance (e.g. vertical position) is sampled more often
		# "sample-period-longest.sim",  # increased performance sampling period so distance and velocity are measured rectilinearly
	]


	# This function is not needed because in python, "For efficiency reasons, each module is only imported once per interpreter session."
	# @staticmethod
	# def getFramsModuleInstance():
	#	"""If some other party needs access to the frams module to directly access or modify Framsticks objects,
	#	use this function to avoid importing the "frams" module multiple times and avoid potentially initializing
	#	it many times."""
	#	return frams

	def __init__(self, frams_path, frams_lib_name, sim_settings_files):
		if frams_lib_name is None:
			frams.init(frams_path)  # could add support for setting alternative directories using -D and -d
		else:
			frams.init(frams_path, "-L" + frams_lib_name)  # could add support for setting alternative directories using -D and -d

		print('Available objects:', dir(frams))
		print()

		print('Performing a basic test 1/2... ', end='')
		simplest = self.getSimplest("1")
		assert simplest == "X" and type(simplest) is str
		print('OK.')
		print('Performing a basic test 2/2... ', end='')
		assert self.isValid(["X[0:0],", "X[0:0]", "X[1:0]"]) == [False, True, False]
		print('OK.')
		if not self.DETERMINISTIC:
			frams.Math.randomize()
		frams.Simulator.expdef = "standard-eval"  # this expdef (or fully compatible) must be used by EVALUATION_SETTINGS_FILE
		if sim_settings_files is not None:
			self.EVALUATION_SETTINGS_FILE = sim_settings_files
		print('Using settings:', self.EVALUATION_SETTINGS_FILE)
		assert isinstance(self.EVALUATION_SETTINGS_FILE, list)  # ensure settings file(s) are provided as a list
		for simfile in self.EVALUATION_SETTINGS_FILE:
			frams.Simulator.ximport(simfile, 4 + 8 + 16)


	def getSimplest(self, genetic_format) -> str:
		return frams.GenMan.getSimplest(genetic_format).genotype._string()


	def evaluate(self, genotype_list: List[str]):
		"""
		Returns:
			List of dictionaries containing the performance of genotypes evaluated using self.EVALUATION_SETTINGS_FILE.
			Note that for whatever reason (e.g. incorrect genotype), the dictionaries you will get may be empty or
			partially empty and may not have the fields you expected, so handle such cases properly.
		"""
		assert isinstance(genotype_list, list)  # because in python, str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		if not self.PRINT_FRAMSTICKS_OUTPUT:
			ec = frams.MessageCatcher.new()  # mute potential errors, warnings, messages

		frams.GenePools[0].clear()
		for g in genotype_list:
			frams.GenePools[0].add(g)
		frams.ExpProperties.evalsavefile = ""  # no need to store results in a file - we will get evaluations directly from Genotype's "data" field
		frams.Simulator.init()
		frams.Simulator.start()

		# step = frams.Simulator.step  # cache reference to avoid repeated lookup in the loop (just for performance)
		# while frams.Simulator.running._int():  # standard-eval.expdef sets running to 0 when the evaluation is complete
		#	step()
		frams.Simulator.eval("while(Simulator.running) Simulator.step();")  # fastest
		# Timing for evaluating a single simple creature 100x:
		# - python step without caching: 2.2s
		# - python step with caching   : 1.6s
		# - pure FramScript and eval() : 0.4s

		if not self.PRINT_FRAMSTICKS_OUTPUT:
			if ec.error_count._value() > 0:  # errors are important and should not be ignored, at least display how many
				print("[ERROR]", ec.error_count, "error(s) and", ec.warning_count, "warning(s) while evaluating", len(genotype_list), "genotype(s)")
			ec.close()

		results = []
		for g in frams.GenePools[0]:
			serialized_dict = frams.String.serialize(g.data[frams.ExpProperties.evalsavedata._value()])
			evaluations = json.loads(serialized_dict._string())  # Framsticks native ExtValue's get converted to native python types such as int, float, list, str.
			# now, for consistency with FramsticksCLI.py, add "num" and "name" keys that are missing because we got data directly from Genotype, not from the file produced by standard-eval.expdef's function printStats(). What we do below is what printStats() does.
			result = {"num": g.num._value(), "name": g.name._value(), "evaluations": evaluations}
			results.append(result)

		return results


	def mutate(self, genotype_list: List[str]) -> List[str]:
		"""
		Returns:
			The genotype(s) of the mutated source genotype(s). self.GENOTYPE_INVALID for genotypes whose mutation failed (for example because the source genotype was invalid).
		"""
		assert isinstance(genotype_list, list)  # because in python, str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		mutated = []
		for g in genotype_list:
			mutated.append(frams.GenMan.mutate(frams.Geno.newFromString(g)).genotype._string())
		assert len(genotype_list) == len(mutated), "Submitted %d genotypes, received %d validity values" % (len(genotype_list), len(mutated))
		return mutated


	def crossOver(self, genotype_parent1: str, genotype_parent2: str) -> str:
		"""
		Returns:
			The genotype of the offspring. self.GENOTYPE_INVALID if the crossing over failed.
		"""
		return frams.GenMan.crossOver(frams.Geno.newFromString(genotype_parent1), frams.Geno.newFromString(genotype_parent2)).genotype._string()


	def dissimilarity(self, genotype_list: List[str], method: int) -> np.ndarray:
		"""
			:param method: -1 = genetic Levenshtein distance; 0, 1, 2 = phenetic dissimilarity (SimilMeasureGreedy, SimilMeasureHungarian, SimilMeasureDistribution)
			:return: A square array with dissimilarities of each pair of genotypes.
		"""
		assert isinstance(genotype_list, list)  # because in python, str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity

		# if you want to override what EVALUATION_SETTINGS_FILE sets, you can do it below:
		# frams.SimilMeasureHungarian.simil_partgeom = 1
		# frams.SimilMeasureHungarian.simil_weightedMDS = 1

		n = len(genotype_list)
		square_matrix = np.zeros((n, n))

		if method in (0, 1, 2):  # Framsticks phenetic dissimilarity methods
			frams.SimilMeasure.simil_type = method
			genos = []  # prepare an array of Geno objects so that we don't need to convert raw strings to Geno objects all the time in loops
			for g in genotype_list:
				genos.append(frams.Geno.newFromString(g))
			frams_evaluateDistance = frams.SimilMeasure.evaluateDistance  # cache function reference for better performance in loops
			for i in range(n):
				for j in range(n):  # maybe calculate only one triangle if you really need a 2x speedup
					square_matrix[i][j] = frams_evaluateDistance(genos[i], genos[j])._double()
		elif method == -1:
			import Levenshtein
			for i in range(n):
				for j in range(n):  # maybe calculate only one triangle if you really need a 2x speedup
					square_matrix[i][j] = Levenshtein.distance(genotype_list[i], genotype_list[j])
		else:
			raise Exception("Don't know what to do with dissimilarity method = %d" % method)

		for i in range(n):
			assert square_matrix[i][i] == 0, "Not a correct dissimilarity matrix, diagonal expected to be 0"
		non_symmetric_diff = square_matrix - square_matrix.T
		non_symmetric_count = np.count_nonzero(non_symmetric_diff)
		if non_symmetric_count > 0:
			non_symmetric_diff_abs = np.abs(non_symmetric_diff)
			max_pos1d = np.argmax(non_symmetric_diff_abs)  # location of the largest discrepancy
			max_pos2d_XY = np.unravel_index(max_pos1d, non_symmetric_diff_abs.shape)  # 2D coordinates of the largest discrepancy
			max_pos2d_YX = max_pos2d_XY[1], max_pos2d_XY[0]  # 2D coordinates of the largest discrepancy mirror
			worst_guy_XY = square_matrix[max_pos2d_XY]  # this distance and the other below (its mirror) are most different
			worst_guy_YX = square_matrix[max_pos2d_YX]
			print("[WARN] Dissimilarity matrix: expecting symmetry, but %g out of %d pairs were asymmetrical, max difference was %g (%g %%)" %
			      (non_symmetric_count / 2,
			       n * (n - 1) / 2,
			       non_symmetric_diff_abs[max_pos2d_XY],
			       non_symmetric_diff_abs[max_pos2d_XY] * 100 / ((worst_guy_XY + worst_guy_YX) / 2)))  # max diff is not necessarily max %
		return square_matrix


	def isValid(self, genotype_list: List[str]) -> List[bool]:
		assert isinstance(genotype_list, list)  # because in python, str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		valid = []
		for g in genotype_list:
			valid.append(frams.Geno.newFromString(g).is_valid._int() == 1)
		assert len(genotype_list) == len(valid), "Tested %d genotypes, received %d validity values" % (len(genotype_list), len(valid))
		return valid


def parseArguments():
	parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to the Framsticks library (.dll or .so) without trailing slash.')
	parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
	parser.add_argument('-simsettings', required=False, help='The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, "eval-allcriteria.sim" is assumed by default. Must be compatible with the "standard-eval" expdef.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or S. If not given, f1 is assumed.')
	return parser.parse_args()


def ensureDir(string):
	if os.path.isdir(string):
		return string
	else:
		raise NotADirectoryError(string)


if __name__ == "__main__":
	# A demo run.

	# TODO ideas:
	# - check_validity with three levels (invalid, corrected, valid)
	# - a pool of binaries running simultaneously, balance load - in particular evaluation

	parsed_args = parseArguments()
	framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.simsettings)

	print("Sending a direct command to Framsticks library that calculates \"4\"+2 yields", frams.Simulator.eval("return \"4\"+2;"))

	simplest = framsLib.getSimplest('1' if parsed_args.genformat is None else parsed_args.genformat)
	print("\tSimplest genotype:", simplest)
	parent1 = framsLib.mutate([simplest])[0]
	parent2 = parent1
	MUTATE_COUNT = 10
	for x in range(MUTATE_COUNT):  # example of a chain of 10 mutations
		parent2 = framsLib.mutate([parent2])[0]
	print("\tParent1 (mutated simplest):", parent1)
	print("\tParent2 (Parent1 mutated %d times):" % MUTATE_COUNT, parent2)
	offspring = framsLib.crossOver(parent1, parent2)
	print("\tCrossover (Offspring):", offspring)
	print('\tDissimilarity of Parent1 and Offspring:', framsLib.dissimilarity([parent1, offspring], 1)[0, 1])
	print('\tPerformance of Offspring:', framsLib.evaluate([offspring]))
	print('\tValidity of Parent1, Parent 2, and Offspring:', framsLib.isValid([parent1, parent2, offspring]))
