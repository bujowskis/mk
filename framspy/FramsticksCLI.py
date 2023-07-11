from subprocess import Popen, PIPE, check_output
from enum import Enum
from typing import List  # to be able to specify a type hint of list(something)
from itertools import count  # for tracking multiple instances
import json
import sys, os
import argparse
import numpy as np
from framsfiles import reader as framsreader
from framsfiles import writer as framswriter


class FramsticksCLI:
	"""Note: instead of this class, you should use the simpler, faster, and more reliable FramsticksLib.py.
	
	This class runs Framsticks CLI (command-line) executable and communicates with it using standard input and output.
	You can perform basic operations like mutation, crossover, and evaluation of genotypes.
	This way you can perform evolution controlled by python as well as access and manipulate genotypes.
	You can even design and use in evolution your own genetic representation implemented entirely in python.

	You need to provide one or two parameters when you run this class: the path to Framsticks CLI
	and, optionally, the name of the Framsticks CLI executable (if it is non-standard). See::
		FramsticksCLI.py -h"""

	PRINT_FRAMSTICKS_OUTPUT: bool = False  # set to True for debugging
	DETERMINISTIC: bool = False  # set to True to have the same results in each run

	GENO_SAVE_FILE_FORMAT = Enum('GENO_SAVE_FILE_FORMAT', 'NATIVEFRAMS RAWGENO')  # how to save genotypes
	OUTPUT_DIR = "scripts_output"
	GENOTYPE_INVALID = "/*invalid*/"  # this is how genotype invalidity is represented in Framsticks
	STDOUT_ENDOPER_MARKER = "FileObject.write:"  # we look for this message on Framsticks CLI stdout to detect when Framsticks created a file with the result we expect

	FILE_PREFIX = 'framspy_'

	RANDOMIZE_CMD = "Math.randomize();"
	SETEXPEDEF_CMD = "Simulator.expdef=\"standard-eval\";"
	GETSIMPLEST_CMD = "getsimplest"
	GETSIMPLEST_FILE = "simplest.gen"
	EVALUATE_CMD = "evaluate eval-allcriteria.sim"  # the .sim file must be compatible with the standard-eval expdef
	EVALUATE_FILE = "genos_eval.json"
	CROSSOVER_CMD = "crossover"
	CROSSOVER_FILE = "crossover_child.gen"
	DISSIMIL_CMD = "dissimil"
	DISSIMIL_FILE = "dissimilarity_matrix.tsv"  # tab-separated values
	ISVALID_CMD = "isvalid_many"
	ISVALID_FILE = "validity.txt"
	MUTATE_CMD = "mutate_many"
	MUTATE_FILE = "mutation_results.gen"

	CLI_INPUT_FILE = "genotypes.gen"

	_next_instance_id = count(0)  # "static" counter incremented when a new instance is created. Used to ensure unique filenames for each instance.


	def __init__(self, framspath, framsexe, pid=""):
		self.pid = pid if pid is not None else ""
		self.id = next(FramsticksCLI._next_instance_id)
		self.frams_path = framspath
		self.frams_exe = framsexe if framsexe is not None else 'frams.exe' if os.name == "nt" else 'frams.linux'
		self.writing_path = None
		mainpath = os.path.join(self.frams_path, self.frams_exe)
		exe_call = [mainpath, '-Q', '-s', '-c', '-icliutils.ini']  # -c will be ignored in Windows Framsticks (this option is meaningless because the Windows version does not support color console, so no need to deactivate this feature using -c)
		exe_call_to_get_version = [mainpath, '-V']
		exe_call_to_get_path = [mainpath, '-?']
		try:
			print("\n".join(self.__readAllOutput(exe_call_to_get_version)))
			help = self.__readAllOutput(exe_call_to_get_path)
			for helpline in help:
				if 'dDIRECTORY' in helpline:
					self.writing_path = helpline.split("'")[1]
		except FileNotFoundError:
			print("Could not find Framsticks executable ('%s') in the given location ('%s')." % (self.frams_exe, self.frams_path))
			sys.exit(1)
		print("Temporary files with results will be saved in detected writable working directory '%s'" % self.writing_path)
		self.__spawnFramsticksCLI(exe_call)


	def __readAllOutput(self, command):
		frams_process = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
		return [line.decode('utf-8').rstrip() for line in iter(frams_process.stdout.readlines())]


	def __spawnFramsticksCLI(self, args):
		# the child app (Framsticks CLI) should not buffer outputs and we need to immediately read its stdout, hence we use pexpect/wexpect
		print('Spawning Framsticks CLI for continuous stdin/stdout communication... ', end='')
		if os.name == "nt":  # Windows:
			import wexpect  # https://pypi.org/project/wexpect/
			# https://github.com/raczben/wexpect/tree/master/examples
			self.child = wexpect.spawn(' '.join(args))
		else:
			import pexpect  # https://pexpect.readthedocs.io/en/stable/
			self.child = pexpect.spawn(' '.join(args))
		self.child.setecho(False)  # ask the communication to not copy to stdout what we write to stdin
		print('OK.')

		self.__readFromFramsCLIUntil("UserScripts.autoload")
		print('Performing a basic test 1/2... ', end='')
		assert self.getSimplest("1") == "X"
		print('OK.')
		print('Performing a basic test 2/2... ', end='')
		assert self.isValid(["X[0:0],", "X[0:0]", "X[1:0]"]) == [False, True, False]
		print('OK.')
		if not self.DETERMINISTIC:
			self.sendDirectCommand(self.RANDOMIZE_CMD)
		self.sendDirectCommand(self.SETEXPEDEF_CMD)


	def closeFramsticksCLI(self):
		# End gracefully by sending end-of-file character: ^Z or ^D
		# Without the -Q argument ("quiet mode"), Framsticks CLI would print "Shell closed." for goodbye.
		self.child.sendline(chr(26 if os.name == "nt" else 4))


	def __getPrefixedFilename(self, filename: str) -> str:
		# Returns filename with unique instance id appended so there is no clash when many instances of this class use the same Framsticks CLI executable
		return FramsticksCLI.FILE_PREFIX + self.pid + str(chr(ord('A') + self.id)) + '_' + filename


	def __saveGenotypeToFile(self, genotype, name, mode, saveformat):
		relname = self.__getPrefixedFilename(name)
		absname = os.path.join(self.writing_path, relname)
		if mode == 'd':  # special mode, 'delete'
			if os.path.exists(absname):
				os.remove(absname)
		else:
			outfile = open(absname, mode)
			if saveformat == self.GENO_SAVE_FILE_FORMAT["RAWGENO"]:
				outfile.write(genotype)
			else:
				outfile.write(framswriter.from_collection({"_classname": "org", "genotype": genotype}))
				outfile.write("\n")
			outfile.close()
		return relname, absname


	def __readFromFramsCLIUntil(self, until_marker: str) -> str:
		output = ""
		while True:
			self.child.expect('\r\n' if os.name == "nt" else '\n')
			msg = str(self.child.before)
			if self.PRINT_FRAMSTICKS_OUTPUT or msg.startswith("[ERROR]") or msg.startswith("[CRITICAL]"):
				print(msg)
			if until_marker in msg:
				break
			else:
				output += msg + '\n'
		return output


	def __runCommand(self, command, genotypes, result_file_name, saveformat) -> List[str]:
		filenames_rel = []  # list of file names with input data for the command
		filenames_abs = []  # same list but absolute paths actually used
		if saveformat == self.GENO_SAVE_FILE_FORMAT["RAWGENO"]:
			for i in range(len(genotypes)):
				# plain text format = must have a separate file for each genotype
				rel, abs = self.__saveGenotypeToFile(genotypes[i], "genotype" + str(i) + ".gen", "w", self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
				filenames_rel.append(rel)
				filenames_abs.append(abs)
		elif saveformat == self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"]:
			self.__saveGenotypeToFile(None, self.CLI_INPUT_FILE, 'd', None)  # 'd'elete: ensure there is nothing left from the last run of the program because we "a"ppend to file in the loop below
			for i in range(len(genotypes)):
				rel, abs = self.__saveGenotypeToFile(genotypes[i], self.CLI_INPUT_FILE, "a", self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
			#  since we use the same file in the loop above, add this file only once (i.e., outside of the loop)
			filenames_rel.append(rel)
			filenames_abs.append(abs)

		result_file_name = self.__getPrefixedFilename(result_file_name)
		cmd = command + " " + " ".join(filenames_rel) + " " + result_file_name
		self.child.sendline(cmd)
		self.__readFromFramsCLIUntil(self.STDOUT_ENDOPER_MARKER)
		filenames_abs.append(os.path.join(self.writing_path, self.OUTPUT_DIR, result_file_name))
		return filenames_abs  # last element is a path to the file containing results


	def __cleanUpCommandResults(self, filenames):
		"""Deletes files with results just created by the command."""
		for name in filenames:
			os.remove(name)


	sendDirectCommand_counter = count(0)  # an internal counter for the sendDirectCommand() method; should be static within that method but python does not allow


	def sendDirectCommand(self, command: str) -> str:
		"""Sends any command to Framsticks CLI. Use when you know Framsticks and its scripting language, Framscript.

		Returns:
			The output of the command, likely with extra \\n because for each entered command, Framsticks CLI responds with a (muted in Quiet mode) prompt and a \\n.
		"""
		self.child.sendline(command.strip())
		next(FramsticksCLI.sendDirectCommand_counter)
		STDOUT_ENDOPER_MARKER = "uniqe-marker-" + str(FramsticksCLI.sendDirectCommand_counter)
		self.child.sendline("Simulator.print(\"%s\");" % STDOUT_ENDOPER_MARKER)
		return self.__readFromFramsCLIUntil(STDOUT_ENDOPER_MARKER)


	def getSimplest(self, genetic_format) -> str:
		files = self.__runCommand(self.GETSIMPLEST_CMD + " " + genetic_format + " ", [], self.GETSIMPLEST_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			genotype = "".join(f.readlines())
		self.__cleanUpCommandResults(files)
		return genotype


	def evaluate(self, genotype_list: List[str]):
		"""
		Returns:
			List of dictionaries containing the performance of genotypes evaluated with self.EVALUATE_COMMAND.
			Note that for whatever reason (e.g. incorrect genotype), the dictionaries you will get may be empty or
			partially empty and may not have the fields you expected, so handle such cases properly.
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		files = self.__runCommand(self.EVALUATE_CMD, genotype_list, self.EVALUATE_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		with open(files[-1]) as f:
			data = json.load(f)
		if len(data) > 0:
			self.__cleanUpCommandResults(files)
			assert len(genotype_list) == len(data), f"After evaluating {len(genotype_list)} genotype(s) got {len(data)} result(s)."
			return data
		else:
			print("Evaluating genotype: no performance data was returned in", self.EVALUATE_FILE)  # we do not delete files here
			return None


	def mutate(self, genotype_list: List[str]) -> List[str]:
		"""
		Returns:
			The genotype(s) of the mutated source genotype(s). self.GENOTYPE_INVALID for genotypes whose mutation failed (for example because the source genotype was invalid).
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		files = self.__runCommand(self.MUTATE_CMD, genotype_list, self.MUTATE_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		genos = framsreader.load(files[-1], "gen file")
		self.__cleanUpCommandResults(files)
		return [g["genotype"] for g in genos]


	def crossOver(self, genotype_parent1: str, genotype_parent2: str) -> str:
		"""
		Returns:
			The genotype of the offspring. self.GENOTYPE_INVALID if the crossing over failed.
		"""
		files = self.__runCommand(self.CROSSOVER_CMD, [genotype_parent1, genotype_parent2], self.CROSSOVER_FILE, self.GENO_SAVE_FILE_FORMAT["RAWGENO"])
		with open(files[-1]) as f:
			child_genotype = "".join(f.readlines())
		self.__cleanUpCommandResults(files)
		return child_genotype


	def dissimilarity(self, genotype_list: List[str]) -> np.ndarray:
		"""
		Returns:
			A square array with dissimilarities of each pair of genotypes.
		"""
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		files = self.__runCommand(self.DISSIMIL_CMD, genotype_list, self.DISSIMIL_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		with open(files[-1]) as f:
			dissimilarity_matrix = np.genfromtxt(f, dtype=np.float64, comments='#', encoding=None, delimiter='\t')
		# We would like to skip column #1 while reading and read everything else, but... https://stackoverflow.com/questions/36091686/exclude-columns-from-genfromtxt-with-numpy
		# This would be too complicated, so strings (names) in column #1 become NaN as floats (unless they accidentally are valid numbers) - not great, not terrible
		square_matrix = dissimilarity_matrix[:, 2:]  # get rid of two first columns (fitness and name)
		EXPECTED_SHAPE = (len(genotype_list), len(genotype_list))
		# print(square_matrix)
		assert square_matrix.shape == EXPECTED_SHAPE, f"Not a correct dissimilarity matrix, expected {EXPECTED_SHAPE}"
		for i in range(len(square_matrix)):
			assert square_matrix[i][i] == 0, "Not a correct dissimilarity matrix, diagonal expected to be 0"
		assert (square_matrix == square_matrix.T).all(), "Probably not a correct dissimilarity matrix, expecting symmetry, verify this"  # could introduce tolerance in comparison (e.g. class field DISSIMIL_DIFF_TOLERANCE=10^-5) so that miniscule differences do not fail here
		self.__cleanUpCommandResults(files)
		return square_matrix


	def isValid(self, genotype_list: List[str]) -> List[bool]:
		assert isinstance(genotype_list, list)  # because in python str has similar capabilities as list and here it would pretend to work too, so to avoid any ambiguity
		files = self.__runCommand(self.ISVALID_CMD, genotype_list, self.ISVALID_FILE, self.GENO_SAVE_FILE_FORMAT["NATIVEFRAMS"])
		valid = []
		with open(files[-1]) as f:
			for line in f:
				valid.append(line.strip() == "1")
		self.__cleanUpCommandResults(files)
		assert len(genotype_list) == len(valid), "Submitted %d genotypes, received %d validity values" % (len(genotype_list), len(valid))
		return valid


def parseArguments():
	parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
	parser.add_argument('-exe', required=False, help='Executable name. If not given, "frams.exe" or "frams.linux" is assumed depending on the platform.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or S. If not given, f1 is assumed.')
	parser.add_argument('-pid', required=False, help='Unique ID of this process. Only relevant when you run multiple instances of this class simultaneously but as separate processes, and they use the same Framsticks CLI executable. This value will be appended to the names of created files to avoid conflicts.')
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
	# - "vectorize" crossover so that many genotypes is handled in one call. Even better, use .so/.dll direct communication to CLI
	# - use threads for non-blocking reading from frams' stdout and thus not relying on specific strings printed by frams
	# - a pool of binaries running simultaneously, balance load - in particular, evaluation

	parsed_args = parseArguments()
	framsCLI = FramsticksCLI(parsed_args.path, parsed_args.exe, parsed_args.pid)

	print("Sending a direct command to Framsticks CLI that calculates \"4\"+2 yields", repr(framsCLI.sendDirectCommand("Simulator.print(\"4\"+2);")))

	simplest = framsCLI.getSimplest('1' if parsed_args.genformat is None else parsed_args.genformat)
	print("\tSimplest genotype:", simplest)
	parent1 = framsCLI.mutate([simplest])[0]
	parent2 = parent1
	MUTATE_COUNT = 10
	for x in range(MUTATE_COUNT):  # example of a chain of 10 mutations
		parent2 = framsCLI.mutate([parent2])[0]
	print("\tParent1 (mutated simplest):", parent1)
	print("\tParent2 (Parent1 mutated %d times):" % MUTATE_COUNT, parent2)
	offspring = framsCLI.crossOver(parent1, parent2)
	print("\tCrossover (Offspring):", offspring)
	print('\tDissimilarity of Parent1 and Offspring:', framsCLI.dissimilarity([parent1, offspring])[0, 1])
	print('\tPerformance of Offspring:', framsCLI.evaluate([offspring]))
	print('\tValidity of Parent1, Parent 2, and Offspring:', framsCLI.isValid([parent1, parent2, offspring]))

	framsCLI.closeFramsticksCLI()
