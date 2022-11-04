import pickle
import os
from abc import ABC, abstractmethod
from tkinter import W
from ..base.random_sequence_index import RandomIndexSequence
from ..structures.individual import Individual

BAD_FITNESS = None
STATS_SAVE_ONLY_BEST_FITNESS = True


class ExperimentABC(ABC):
    current_population = []
    hof = []
    stats = []
    current_genneration = 0

    def select(self, individuals, tournament_size, random_index_sequence):
        """
        Tournament selection, returns the index of the best individual from those taking part in the tournament
        """
        best_index = None
        for i in range(tournament_size):
            rnd_index = random_index_sequence.getNext()
            if best_index is None or individuals[rnd_index].fitness > best_index.fitness:
                best_index = individuals[rnd_index]
        return best_index

    def addGenotypeIfValid(self, ind_list, genotype):
            new_individual = Individual(self.evaluate)
            new_individual.setAndEvaluate(genotype)
            if new_individual.fitness is not None:  # this is how we defined BAD_FITNESS in evaluate()
                ind_list.append(new_individual)

    def make_new_population(self, individuals, prob_mut, prob_xov, tournament_size):
        """
        'individuals' is the input population (a list of individuals).
        Assumptions: all genotypes in 'individuals' are valid and evaluated (have fitness set).
        Returns: a new population of the same size as 'individuals' with prob_mut mutants, prob_xov offspring, and the remainder of clones.
        """
        newpop = []
        N = len(individuals)
        expected_mut = int(N * prob_mut)
        expected_xov = int(N * prob_xov)
        assert expected_mut + expected_xov <= N, "If probabilities of mutation (%g) and crossover (%g) added together exceed 1.0, then the population would grow every generation..." % (prob_mut, prob_xov)
        ris = RandomIndexSequence(N)

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
        while len(newpop) < len(individuals):
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            newpop.append(Individual(self.evaluate).copyFrom(ind))

        return newpop

    def save_state(self, state_filename):
        state = self.get_state()
        if state_filename is None:
            return
        state_filename_tmp = state_filename + ".tmp"
        try:
            with open(state_filename_tmp, "wb") as f:
                pickle.dump(state, f)
            os.replace(state_filename_tmp, state_filename)  # ensures the new file was first saved OK (e.g. enough free space on device), then replace
        except Exception as ex:
            raise RuntimeError("Failed to save evolution state '%s' (because: %s). This does not prevent the experiment from continuing, but let's stop here to fix the problem with saving state files." % (state_filename_tmp, ex))

    def load_state(self, state_filename):
        if state_filename is None:
            print("File name not provided")
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
        return [self.current_genneration,self.current_population,self.stats]

    def set_state(self,state):
        self.current_genneration,self.current_population,self.stats = state

    def update_stats(self, generation, all_individuals):
        worst = min(all_individuals, key=lambda item: item.rawfitness)
        best = max(all_individuals, key=lambda item: item.rawfitness)
        self.hof.add(best)  # instead of single best, could add all individuals in population here, but then the outcome would depend on the order of adding
        self.stats.append(best.rawfitness if STATS_SAVE_ONLY_BEST_FITNESS else best)
        print("%d\t%d\t%g\t%g" % (generation, len(all_individuals), worst.rawfitness, best.rawfitness))

    @abstractmethod
    def mutate(self, gen1):
        pass

    @abstractmethod
    def cross_over(self, gen1, gen2):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass

    @abstractmethod
    def evolve(self):
        pass
