from pandas import DataFrame
import time
import numpy as np
import copy
import json
from evolalg.json_encoders import Encoder

from evolalg.hfc_base.experiment_hfc import ExperimentHFC
from evolalg.utils import get_state_filename, evaluate_cec2017
from evolalg.structures.population_methods import reinitialize_population_with_random_numerical, fill_population_with_random_numerical
from evolalg.mutation import cec2017_numerical_mutation
from evolalg.crossover import cec2017_numerical_crossover

from evolalg.base.random_sequence_index import RandomIndexSequence
from evolalg.structures.individual import Individual
from evolalg.constants import BAD_FITNESS


class ExperimentNumericalHFC(ExperimentHFC):
    def __init__(
            self, popsize, hof_size, number_of_populations, migration_interval, save_only_best,
            benchmark_function, dimensions: int, results_directory_path: str
    ):
        super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.benchmark_function = benchmark_function
        self.dimensions = dimensions
        self.results_directory_path = results_directory_path
        self.number_of_epochs: int = None
        self.current_epoch: int = None

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
    ):
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        self.number_of_epochs: int = int(np.floor(generations/self.migration_interval) + 1)  # account for epoch 0 (before start of migration)
        self.current_epoch: int = 0
        time0 = time.process_time()
        
        for pop_idx in range(len(self.populations)):
            self.populations[pop_idx] = reinitialize_population_with_random_numerical(
                population=self.populations[pop_idx],
                dimensions=self.dimensions,
                evaluate=self.evaluate
            )
            for i in self.populations[pop_idx].population:
                i.innovation_in_time = [0.0 for _ in range(self.number_of_epochs)]
                i.innovation_in_time[self.current_epoch] = 1.0

        df = DataFrame(columns=['generation', 'total_popsize', 'worst_fitness', 'best_fitness'])

        # CALIBRATION STAGE
        pool_of_all_individuals = []
        [pool_of_all_individuals.extend(p.population) for p in self.populations]
        fitnesses_of_individuals = [individual.fitness for individual in pool_of_all_individuals]
        self.admission_thresholds[0] = -np.inf
        # EXAMPLE - Passing parameters for HFC-ADM
        self.admission_thresholds[1], self.admission_thresholds[-1] = self.get_bounds(
            pool_of_all_individuals, fitnesses_of_individuals,
            set_worst_to_fixed=True, set_best_to_stdev=True
        )
        lower_bound = self.admission_thresholds[1]
        upper_bound = self.admission_thresholds[-1]
        population_width = (upper_bound - lower_bound) / self.number_of_populations - 2
        for i in range(2, self.number_of_populations):
            self.admission_thresholds[i] = lower_bound + (i - 1) * population_width

        time0 = time.process_time()
        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.current_epoch += 1
                self.migrate_populations()

            pool_of_all_individuals = []
            [pool_of_all_individuals.extend(p.population) for p in self.populations]
            self.update_stats(g, pool_of_all_individuals)
            cli_stats = self.get_cli_stats()
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2], cli_stats[3]]
            self.update_stats(g, pool_of_all_individuals)
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)
            
        return self.hof, self.stats, df

    def make_new_population(self, individuals, prob_mut, prob_xov, tournament_size):
        newpop = []
        expected_mut = int(self.popsize * prob_mut)
        expected_xov = int(self.popsize * prob_xov)
        assert expected_mut + expected_xov <= self.popsize, f"If probabilities of mutation ({prob_mut}) and crossover ({prob_xov}) added together exceed 1.0, then the population would grow every generation..."
        ris = RandomIndexSequence(len(individuals))

        # adding valid mutants of selected individuals...
        while len(newpop) < expected_mut:
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            new_individual = Individual()
            new_individual.set_and_evaluate(self.mutate(ind.genotype), self.evaluate)
            if new_individual.fitness is not BAD_FITNESS:
                new_individual.innovation_in_time = copy.deepcopy(ind.innovation_in_time)
                newpop.append(new_individual)

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind2 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            new_individual = Individual()
            new_individual.set_and_evaluate(self.cross_over(ind1.genotype, ind2.genotype), self.evaluate)
            if new_individual.fitness is not BAD_FITNESS:
                new_individual.innovation_in_time = list(np.average([ind1.innovation_in_time, ind2.innovation_in_time], axis=0))
                newpop.append(new_individual)

        # FIXME - no way to introduce random individuals in make_new_population
        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < self.popsize:
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind_copy = Individual().copyFrom(ind)
            ind_copy.innovation_in_time = copy.deepcopy(ind.innovation_in_time)
            newpop.append(ind_copy)

        return newpop

    def add_to_worst(self):
        self.populations[0] = fill_population_with_random_numerical(
            population=self.populations[0],
            dimensions=self.dimensions,
            evaluate=self.evaluate
        )
        for i in self.populations[0].population:
            if not hasattr(i, 'innovation_in_time'):
                i.innovation_in_time = [0.0 for _ in range(self.number_of_epochs)]
                i.innovation_in_time[self.current_epoch] = 1.0

    def save_genotypes(self, filename):
        state_to_save = {
            "number_of_generations": self.current_generation,
            "hof": [{
                "genotype": individual.genotype,
                "fitness": individual.rawfitness,
                "innovation_in_time": individual.innovation_in_time
            } for individual in self.hof.hof],
        }
        with open(f"{filename}.json", 'w') as f:
            json.dump(state_to_save, f, cls=Encoder)

    def evaluate(self, genotype):
        return evaluate_cec2017(genotype, self.benchmark_function)

    def mutate(self, gen1):
        return cec2017_numerical_mutation(gen1)

    def cross_over(self, gen1, gen2):
        return cec2017_numerical_crossover(gen1, gen2)
