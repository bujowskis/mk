import time
from abc import ABC, abstractmethod
from tkinter import W

import numpy as np
from deap import base, tools
from deap.tools.emo import assignCrowdingDist

from ..constants import BAD_FITNESS
from ..structures.individual import Individual
from .experiment_abc import ExperimentABC
from .remove_diagonal import remove_diagonal


class DeapFitness(base.Fitness):
    weights = (1, 1)

    def __init__(self, *args, **kwargs):
        super(DeapFitness, self).__init__(*args, **kwargs)


class ExperimentNiching(ExperimentABC, ABC):
    fit: str = "niching"
    normalize: str = "None"
    archive_size: int = None

    def __init__(self, fit, normalize, popsize, hof_size, save_only_best=True, knn_niching=5, knn_nslc=10, archive_size=0) -> None:
        ExperimentABC.__init__(self,popsize=popsize, hof_size=hof_size, save_only_best=save_only_best)
        self.fit = fit
        self.normalize = normalize
        self.knn_niching = knn_niching
        self.knn_nslc = knn_nslc
        self.archive_size=archive_size
        if popsize < self.knn_niching:
            self.knn_niching = popsize - 2
        if popsize < self.knn_nslc:
            self.knn_nslc = popsize - 2

    def transform_indexes(self, i, index_array):
        return [x+1 if x >= i else x for x in index_array]

    def normalize_dissim(self, dissim_matrix):
        dissim_matrix = remove_diagonal(np.array(dissim_matrix))
        if self.normalize == "none":
            return dissim_matrix
        elif self.normalize == "max":
            divide_by = np.max(dissim_matrix)
        elif self.normalize == "sum":
            divide_by = np.sum(dissim_matrix)
        else:
            raise Exception(f"Wrong normalization method,", self.normalize)
        if divide_by != 0:
            return dissim_matrix/divide_by
        else:
            return dissim_matrix

    def do_niching(self, population_structures):
        population_archive = population_structures.population + population_structures.archive
        dissim_matrix = self.dissimilarity(population_archive)
        if "knn" not in self.fit:
            dissim_list = np.mean(self.normalize_dissim(dissim_matrix), axis=1)
        else:
            dissim_list = np.mean(np.partition(
                self.normalize_dissim(dissim_matrix), self.knn_niching)[:, :self.knn_niching], axis=1)

        if "niching" in self.fit:
            for i, d in zip(population_archive, dissim_list):
                i.fitness = i.rawfitness * d
        elif "novelty" in self.fit:
            for i, d in zip(population_archive, dissim_list):
                i.fitness = d
        else:
            raise Exception("Wrong fit type: ", self.fit,
                            f" chose correct one or implement new behavior")
        population_structures.update_archive(dissim_matrix, population_archive)

    def do_nsga2_dissim(self, population):
        dissim_matrix = self.dissimilarity(population)
        dissim_list = np.mean(self.normalize_dissim(dissim_matrix), axis=1)
        for i, d in zip(population, dissim_list):
            i.fitness = DeapFitness(tuple((d, i.rawfitness)))

    def do_nslc_dissim(self, population):
        dissim_matrix = self.dissimilarity(population)
        normalized_matrix = self.normalize_dissim(dissim_matrix)
        for i in range(len(normalized_matrix)):
            temp_dissim = normalized_matrix[i]
            index_array = np.argpartition(temp_dissim, kth=self.knn_nslc, axis=-1)[:self.knn_nslc]
            dissim_value = np.mean(np.take_along_axis(
                temp_dissim, index_array, axis=-1))
            temp_fitness = population[i].rawfitness
            population_of_most_different = list(
                map(population.__getitem__, self.transform_indexes(i, index_array)))
            temp_ind_fit = sum(
                [1 for ind in population_of_most_different if ind.rawfitness < temp_fitness])
            population[i].fitness = DeapFitness(
                tuple((dissim_value, temp_ind_fit)))

    def make_new_population_nsga2(self, population, prob_mut, prob_xov):
        expected_mut = int(self.popsize * prob_mut)
        expected_xov = int(self.popsize * prob_xov)
        assert expected_mut + expected_xov <= self.popsize, "If probabilities of mutation (%g) and crossover (%g) added together exceed 1.0, then the population would grow every generation..." % (prob_mut, prob_xov)
        assignCrowdingDist(population)
        offspring = tools.selTournamentDCD(population, self.popsize)

        def addGenotypeIfValid(ind_list, genotype):
            new_individual = Individual()
            new_individual.set_and_evaluate(genotype, self.evaluate)
            if new_individual.fitness is not BAD_FITNESS:
                ind_list.append(new_individual)

        counter = 0

        def get_individual(pop, c):
            if c < len(pop):
                ind = pop[c]
                c += 1
                return ind, c
            else:
                c = 0
                ind = pop[c]
                c += 1
                return ind, c

        newpop = []
        while len(newpop) < expected_mut:
            ind, counter = get_individual(offspring, counter)
            addGenotypeIfValid(newpop, self.mutate(ind.genotype))

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1, counter = get_individual(offspring, counter)
            ind2, counter = get_individual(offspring, counter)
            addGenotypeIfValid(newpop, self.cross_over(ind1.genotype, ind2.genotype))

        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < len(population):
            ind, counter = get_individual(offspring, counter)
            newpop.append(Individual().copyFrom(ind))

        pop_offspring = population+newpop
        print(len(pop_offspring))
        if self.fit == "nslc":
            self.do_nslc_dissim(pop_offspring)
        elif self.fit == "nsga2":
            self.do_nsga2_dissim(pop_offspring)
        out_pop = tools.selNSGA2(pop_offspring, len(population))
        return out_pop

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            # saved generation has been completed, start with the next one
            self.current_generation += 1
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.population_structures.population), len(self.hof),
                                                                                                                                                 len(self.stats),  (len(self.population_structures.archive)), self.current_generation, generations))  # self.current_generation (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self.initialize_evolution(self.genformat, initialgenotype)

        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            if self.fit != "raw" and self.fit != "nsga2" and self.fit != "nslc":
                self.do_niching(self.population_structures)

            if type(self.population_structures.population[0].fitness) == DeapFitness:
                self.population_structures.population = self.make_new_population_nsga2(
                    self.population_structures.population, pmut, pxov)
            else:
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

    @staticmethod
    def get_args_for_parser():
        parser = ExperimentABC.get_args_for_parser()
        parser.add_argument("-dissim",type= int, default= "frams",
                   help="Dissimilarity measure type. Availible -2:emd, -1:lev, 1:frams1 (default}, 2:frams2")
        parser.add_argument("-fit",type= str, default= "raw",
                        help="Fitness type, availible  types: niching, novelty, nsga2, nslc and raw (default}")
        parser.add_argument("-archive",type= int, default= 50,
                            help="Maximum archive size")
        parser.add_argument("-normalize",type= str, default= "max",
                            help="What normalization use for dissimilarity matrix, max (default}, sum and none")
        parser.add_argument("-knn",type= int, default= 0,
                        help="Nearest neighbors parameter for local novelty/niching, if knn==0 global is performed.Default:0")
        return parser 
        
    @abstractmethod
    def dissimilarity(self, population: list):
        pass