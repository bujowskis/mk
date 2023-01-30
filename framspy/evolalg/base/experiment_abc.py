import argparse
import json
import os
import pickle
import time
from abc import ABC, abstractmethod

from ..base.random_sequence_index import RandomIndexSequence
from ..constants import BAD_FITNESS
from ..json_encoders import Encoder
from ..structures.hall_of_fame import HallOfFame
from ..structures.individual import Individual
from ..structures.population import PopulationStructures


class ExperimentABC(ABC):

    population_structures = None
    hof = []
    stats = []
    current_generation = 0
    time_elapsed = 0
    

    def __init__(self, popsize, hof_size, save_only_best=True) -> None:
        self.hof = HallOfFame(hof_size)
        self.popsize = popsize
        self.save_only_best = save_only_best

    def select(self, individuals, tournament_size, random_index_sequence):
        """Tournament selection, returns the index of the best individual from those taking part in the tournament"""
        best_index = None
        for i in range(tournament_size):
            rnd_index = random_index_sequence.getNext()
            if best_index is None or individuals[rnd_index].fitness > best_index.fitness:
                best_index = individuals[rnd_index]
        return best_index

    def addGenotypeIfValid(self, ind_list, genotype):
        new_individual = Individual()
        new_individual.set_and_evaluate(genotype, self.evaluate)
        if new_individual.fitness is not BAD_FITNESS:
            ind_list.append(new_individual)

    def make_new_population(self, individuals, prob_mut, prob_xov, tournament_size):
        """'individuals' is the input population (a list of individuals).
        Assumptions: all genotypes in 'individuals' are valid and evaluated (have fitness set).
        Returns: a new population of size 'self.popsize' with prob_mut mutants, prob_xov offspring, and the remainder of clones."""

        newpop = []
        expected_mut = int(self.popsize * prob_mut)
        expected_xov = int(self.popsize * prob_xov)
        assert expected_mut + expected_xov <= self.popsize, "If probabilities of mutation (%g) and crossover (%g) added together exceed 1.0, then the population would grow every generation..." % (prob_mut, prob_xov)
        ris = RandomIndexSequence(len(individuals))

        # adding valid mutants of selected individuals...
        while len(newpop) < expected_mut:
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            self.addGenotypeIfValid(newpop, self.mutate(ind.genotype))

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind2 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            self.addGenotypeIfValid(newpop, self.cross_over(ind1.genotype, ind2.genotype))

        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < self.popsize:
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            newpop.append(Individual().copyFrom(ind))

        return newpop

    def save_state(self, state_filename):
        state = self.get_state()
        if state_filename is None:
            return
        state_filename_tmp = state_filename + ".tmp"
        try:
            with open(state_filename_tmp, "wb") as f:
                pickle.dump(state, f)
            # ensures the new file was first saved OK (e.g. enough free space on device), then replace
            os.replace(state_filename_tmp, state_filename)
        except Exception as ex:
            raise RuntimeError("Failed to save evolution state '%s' (because: %s). This does not prevent the experiment from continuing, but let's stop here to fix the problem with saving state files." % (
                state_filename_tmp, ex))

    def load_state(self, state_filename):
        if state_filename is None:
            # print("Loading evolution state: file name not provided")
            return None
        try:
            with open(state_filename, 'rb') as f:
                state = pickle.load(f)
                self.set_state(state)
        except FileNotFoundError:
            return None
        print("...Loaded evolution state from '%s'" % state_filename)
        return True

    def get_state_filename(self, save_file_name):
        return None if save_file_name is None else save_file_name + '_state.pkl'

    def get_state(self):
        return [self.time_elapsed, self.current_generation, self.population_structures, self.hof, self.stats]

    def set_state(self, state):
        self.time_elapsed, self.current_generation, self.population_structures, hof_, self.stats = state
        # sorting: ensure that we add from worst to best so all individuals are added to HOF
        for h in sorted(hof_, key=lambda x: x.rawfitness):
            self.hof.add(h)

    def update_stats(self, generation, all_individuals):
        worst = min(all_individuals, key=lambda item: item.rawfitness)
        best = max(all_individuals, key=lambda item: item.rawfitness)
        # instead of single best, could add all individuals in population here, but then the outcome would depend on the order of adding
        self.hof.add(best)
        self.stats.append(
            best.rawfitness if self.save_only_best else best)
        print("%d\t%d\t%g\t%g" % (generation, len(
            all_individuals), worst.rawfitness, best.rawfitness))

    def initialize_evolution(self, initialgenotype):
        self.current_generation = 0
        self.time_elapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual()
        initial_individual.set_and_evaluate(initialgenotype, self.evaluate)
        self.hof.add(initial_individual)
        self.stats.append(
            initial_individual.rawfitness if self.save_only_best else initial_individual)
        self.population_structures = PopulationStructures(
            initial_individual=initial_individual, archive_size=0, popsize=self.popsize)

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            # saved generation has been completed, start with the next one
            self.current_generation += 1
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.population_structures.population), len(self.hof),
                                                                                                                                                 len(self.stats),  (len(self.population_structures.archive)), self.current_generation, generations))  # self.current_generation (and g) are 0-based, parsed_args.generations is 1-based

        else:
            self.initialize_evolution(initialgenotype)
        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            self.population_structures.population = self.make_new_population(
                self.population_structures.population, pmut, pxov, tournament_size)
            self.update_stats(g, self.population_structures.population)
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(file_name)
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)
        return self.population_structures.population, self.stats

    @abstractmethod
    def mutate(self, gen1):
        pass

    @abstractmethod
    def cross_over(self, gen1, gen2):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass

    def save_genotypes(self, filename):
        """Implement if you want to save finall genotypes,in default implementation this function is run once at the end of evolution"""
        state_to_save = {
            "number_of_generations": self.current_generation,
            "hof": [{"genotype": individual.genotype,
                     "fitness": individual.rawfitness} for individual in self.hof.hof]}
        with open(f"{filename}.json", 'w') as f:
            json.dump(state_to_save, f, cls=Encoder)

    
    @staticmethod
    def get_args_for_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('-popsize',type= int, default= 50,
                            help="Population size, default: 50.")
        parser.add_argument('-generations',type= int, default= 5,
                                help="Number of generations, default: 5.")
        parser.add_argument('-tournament',type= int, default= 5,
                            help="Tournament size, default: 5.")
        parser.add_argument('-pmut',type= float, default= 0.7,
                        help="Probability of mutation, default: 0.7")
        parser.add_argument('-pxov',type= float, default= 0.2,
                        help="Probability of crossover, default: 0.2")
        parser.add_argument('-hof_size',type= int, default= 10,
                            help="Number of genotypes in Hall of Fame. Default: 10.")
        parser.add_argument('-hof_savefile',type= str, required= False,
                                help= 'If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen. This also activates saving state (checpoint} file and auto-resuming from the saved state, if this file exists.')
        parser.add_argument('-save_only_best',type= bool, default= True, required= False,
                            help="")
        
        return parser