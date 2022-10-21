import time
import numpy as np

from deap import base
from deap import tools
from deap.tools.emo import assignCrowdingDist

from ..structures.individual import Individual
from ..structures.population import PopulationStructures
from ..frams_base.experiment_frams import ExperimentFrams
from ..base.remove_diagonal import remove_diagonal
from .experiment_frams import STATS_SAVE_ONLY_BEST_FITNESS


class DeapFitness(base.Fitness):
        weights = (1, 1)

        def __init__(self, *args, **kwargs):
            super(DeapFitness, self).__init__(*args, **kwargs)


class ExperimentFramsNiching(ExperimentFrams):
    def __init__(self,frams_lib, optimization_criteria, hof_size, popsize, constraints, normalize, dissim, fit, genformat, archive_size) -> None:
        super().__init__(frams_lib, optimization_criteria, hof_size, popsize, constraints)
        self.normalize = normalize
        self.dissim = dissim
        self.fit = fit
        self.genformat = genformat
        self.archive_size = archive_size

    def transform_indexes(self, i,index_array):
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
        dissim_matrix = self.frams_lib.dissimilarity([i.genotype for i in population_archive], self.dissim) # remove itself
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
        dissim_matrix = self.frams_lib.dissimilarity([i.genotype for i in population],self.dissim)
        dissim_list =  np.mean(self.normalize_dissim(dissim_matrix), axis=1)
        for i,d in zip(population,dissim_list):
            i.fitness = DeapFitness(tuple((d,i.rawfitness)))

    def do_nslc_dissim(self, population):
        dissim_matrix = self.frams_lib.dissimilarity([i.genotype for i in population],self.dissim)
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
            addGenotypeIfValid(newpop, self.frams_lib.mutate([ind.genotype])[0])

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1,counter = get_indyvidual(offspring,counter)
            ind2,counter = get_indyvidual(offspring,counter)
            addGenotypeIfValid(newpop, self.frams_lib.crossOver(ind1.genotype, ind2.genotype))

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

    def _initialize_evolution(self, genformat, initialgenotype):
        self.current_genneration = 0
        self.timeelapsed = 0
        self.stats = []  # stores the best individuals, one from each generation
        initial_individual = Individual(self.evaluate)
        initial_individual.setAndEvaluate(self.frams_getsimplest('1' if genformat is None else genformat, initialgenotype))
        self.hof.add(initial_individual)
        self.stats.append(initial_individual.rawfitness if STATS_SAVE_ONLY_BEST_FITNESS else initial_individual)
        self.current_population = PopulationStructures(evaluate=self.evaluate,initial_individual=initial_individual,archive_size=self.archive_size,popsize=self.popsize)
        if self.fit == "nsga2":
            self.do_nsga2_dissim(self.current_population.population)
        if self.fit == "nslc":
            self.do_nslc_dissim(self.current_population.population)

    def evolve(self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size):
        file_name = self.get_state_filename(hof_savefile)
        state = self.load_state(file_name)
        if state is not None:  # loaded state from file
            self.current_genneration += 1  # saved generation has been completed, start with the next one
            print("...Resuming from saved state: population size = %d, hof size = %d, stats size = %d, archive size = %d, generation = %d/%d" % (len(self.current_population.population), len(self.hof), len(self.stats),  (len(self.current_population.archive)),self.current_genneration, generations))  # self.current_genneration (and g) are 0-based, parsed_args.generations is 1-based
        else:
            self._initialize_evolution(self.genformat,initialgenotype)

        time0 = time.process_time()
        for g in range(self.current_genneration, generations):
            if self.fit != "raw" and self.fit !="nsga2" and self.fit !="nslc":
                self.do_niching(self.current_population)

            if type(self.current_population.population[0].fitness) == DeapFitness:
                self.current_population.population = self.make_new_population_nsga2(self.current_population.population, pmut, pxov)
            else:
                self.current_population.population = self.make_new_population(self.current_population.population, pmut, pxov, tournament_size)

            self.update_stats(self.current_population.population)
            
            if hof_savefile is not None:
                self.current_genneration=g
                self.timeelapsed += time.process_time() - time0
                self.save_state(file_name) 

        return self.current_population.population, self.stats
