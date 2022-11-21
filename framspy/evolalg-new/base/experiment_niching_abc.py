from abc import ABC, abstractmethod
from tkinter import W
import numpy as np

from deap import base
from deap import tools
from deap.tools.emo import assignCrowdingDist

from ..structures.individual import Individual
from .experiment_abc import ExperimentABC
from .remove_diagonal import remove_diagonal


BAD_FITNESS = None
STATS_SAVE_ONLY_BEST_FITNESS = True


class DeapFitness(base.Fitness):
        weights = (1, 1)

        def __init__(self, *args, **kwargs):
            super(DeapFitness, self).__init__(*args, **kwargs)


class ExperimentNiching(ExperimentABC, ABC):
    fit: str = "niching"
    normalize: str = "None"

    def transform_indexes(self, i, index_array):
        return  [x+1 if x >= i else x for x in index_array]
        
    def normalize_dissim(self, dissim_matrix):
        dissim_matrix = remove_diagonal(np.array(dissim_matrix))
        if self.normalize == "none":
            return dissim_matrix
        elif self.normalize == "max":
            divide_by = np.max(dissim_matrix)
        elif self.normalize == "sum":
            divide_by = np.sum(dissim_matrix)
        else:
            raise Exception(f"Wrong normalization method,",self.normalize)
        if divide_by != 0:
            return dissim_matrix/divide_by
        else:
            return dissim_matrix

    def do_niching(self, population_structures):
        population_archive = population_structures.population + population_structures.archive
        dissim_matrix =  self.dissimilarity(population_archive)
        if "knn" not in self.fit:
            dissim_list =  np.mean(self.normalize_dissim(dissim_matrix), axis=1)
        else:
            dissim_list = np.mean(np.partition(self.normalize_dissim(dissim_matrix), 5)[:,:5],axis=1)
            
        if "niching" in self.fit:
            for i,d in zip(population_archive,dissim_list):
                i.fitness = i.rawfitness * d
        elif "novelty" in self.fit:
            for i,d in zip(population_archive,dissim_list):
                i.fitness = d
        else:
            raise Exception("Wrong fit type: ",self.fit,f" chose correct one or implement new behaviour")
        population_structures.update_archive(dissim_matrix,population_archive)

    def do_nsga2_dissim(self, population):
        dissim_matrix = self.dissimilarity(population)
        dissim_list =  np.mean(self.normalize_dissim(dissim_matrix), axis=1)
        for i,d in zip(population,dissim_list):
            i.fitness = DeapFitness(tuple((d,i.rawfitness)))

    def do_nslc_dissim(self, population):
        dissim_matrix = self.dissimilarity(population)
        normalized_matrix = self.normalize_dissim(dissim_matrix)
        for i in range(len(normalized_matrix)):
            temp_dissim = normalized_matrix[i]
            index_array = np.argpartition(temp_dissim, kth=30,axis=-1)[:30]
            dissim_value = np.mean(np.take_along_axis(temp_dissim, index_array, axis=-1))
            temp_fitness = population[i].rawfitness
            population_of_most_different = list(map(population.__getitem__, self.transform_indexes(i,index_array)))
            temp_ind_fit = sum([1 for ind in population_of_most_different if ind.rawfitness < temp_fitness])
            population[i].fitness = DeapFitness(tuple((dissim_value,temp_ind_fit)))

    def make_new_population_nsga2(self, population, prob_mut, prob_xov):
        N = len(population)
        expected_mut = int(N * prob_mut)
        expected_xov = int(N * prob_xov)
        assignCrowdingDist(population)
        offspring = tools.selTournamentDCD(population,N)
        
        def addGenotypeIfValid(ind_list, genotype):
            new_individual = Individual(self.evaluate)
            new_individual.setAndEvaluate(genotype)
            if new_individual.fitness is not None:  # this is how we defined BAD_FITNESS in frams_evaluate()
                ind_list.append(new_individual)
        
        counter = 0
        def get_indyvidual(pop,c):
            if c < len(pop):
                ind = pop[c]
                c +=1
                return ind,c
            else:
                c = 0
                ind = pop[c]
                c +=1
                return ind,c
                
        newpop = []
        while len(newpop) < expected_mut:
            ind,counter = get_indyvidual(offspring,counter)
            addGenotypeIfValid(newpop, self.mutate(ind.genotype))

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1,counter = get_indyvidual(offspring,counter)
            ind2,counter = get_indyvidual(offspring,counter)
            addGenotypeIfValid(newpop, self.cross_over(ind1.genotype, ind2.genotype))

        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < len(population):
            ind,counter = get_indyvidual(offspring,counter)
            newpop.append(Individual(self.evaluate).copyFrom(ind))
        
        pop_offspring = population+newpop
        print(len(pop_offspring))
        if self.fit == "nslc":
            self.do_nslc_dissim(pop_offspring)
        elif self.fit == "nsga2":
            self.do_nsga2_dissim(pop_offspring)
        out_pop = tools.selNSGA2(pop_offspring,len(population))
        return out_pop

    @abstractmethod
    def dissimilarity(self, population: list):
        pass