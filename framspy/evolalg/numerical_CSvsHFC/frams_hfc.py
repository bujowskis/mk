from pandas import DataFrame
import time
import numpy as np
import copy
import json
from evolalg.json_encoders import Encoder
from evolalg.structures.population_methods import fill_population_with_random_frams, reinitialize_population_with_random_frams
from evolalg.cs_base.experiment_convection_selection_equiwidth import ExperimentConvectionSelectionEquiwidth
from evolalg.utils import get_state_filename
from evolalg.base.random_sequence_index import RandomIndexSequence
from evolalg.structures.individual import Individual
from ..frams_base.experiment_frams import ExperimentFrams
from ..structures.population import PopulationStructures
from ..base.experiment_islands_model_abc import ExperimentIslands
from evolalg.constants import BAD_FITNESS


class ExperimentFramsCSEquiwidth(ExperimentConvectionSelectionEquiwidth, ExperimentFrams):
    def __init__(self, frams_lib, optimization_criteria, hof_size,
                 popsize, constraints, genformat,
                 number_of_populations, migration_interval, save_only_best,
                 results_directory_path) -> None:
        ExperimentConvectionSelectionEquiwidth.__init__(self, popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        ExperimentFrams.__init__(self, frams_lib=frams_lib, optimization_criteria=optimization_criteria,
                                 hof_size=hof_size, popsize=popsize,
                                 genformat=genformat, save_only_best=save_only_best, constraints=constraints)
        self.number_of_epochs: int = None
        self.current_epoch: int = None
        self.results_directory_path = results_directory_path

    def evolve(
                self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
                try_from_saved_file: bool = True  # to enable in-code disabling of loading saved savefile
        ):
            self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
            self.number_of_epochs: int = int(np.floor(generations/self.migration_interval) + 1)  # account for epoch 0 (before start of migration)
            self.current_epoch: int = 0
            time0 = time.process_time()
            
            for pop_idx in range(len(self.populations)):
                ## TODO 
                self.populations[pop_idx] = reinitialize_population_with_random_frams(
                    population=self.populations[pop_idx],
                    evaluate=self.evaluate
                )
                for i in self.populations[pop_idx].population:
                    i.innovation_in_time = [0.0 for _ in range(self.number_of_epochs)]
                    i.innovation_in_time[self.current_epoch] = 1.0
                    i.contributor_spops = [0.0 for _ in range(self.number_of_populations+1)]
                    # FIXME - same approach as in contributor_spops?
                    i.avg_migration_jump = [0.0 for _ in range(self.number_of_populations*2 + 1)]

            df = DataFrame(columns=['generation', 'total_popsize', 'best_fitness', 'contributor_spops', 'innovation_in_time', 'avg_migration_jump'])

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
                    for idx, population in enumerate(self.populations):
                        for individual in population.population:
                            individual.prev_spop = idx

                    self.migrate_populations()

                    for cur_spop, population in enumerate(self.populations):
                        for individual in population.population:
                            prev_spop_idx = [0.0 for _ in range(self.number_of_populations+1)]
                            # prev_spop_idx[self.current_epoch-1] = 1.0
                            prev_spop_idx[individual.prev_spop] = 1.0
                            individual.contributor_spops = list(
                                ((self.current_epoch-1) * np.array(individual.contributor_spops) + np.array(prev_spop_idx)) / self.current_epoch
                            )
                            # individual.avg_migration_jump = individual.avg_migration_jump + abs(individual.prev_spop - cur_spop)
                            # FIXME - same approach as in contributor_spops?
                            avg_mig = [0.0 for _ in range(self.number_of_populations*2 + 1)]
                            avg_mig[abs(individual.prev_spop - cur_spop)] = 1.0
                            individual.avg_migration_jump = list(
                                ((self.current_epoch-1) * np.array(individual.avg_migration_jump) + np.array(avg_mig)) / self.current_epoch
                            )

                pool_of_all_individuals = []
                [pool_of_all_individuals.extend(p.population) for p in self.populations]
                self.update_stats(g, pool_of_all_individuals)
                cli_stats = self.get_cli_stats()
                df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2], pool_of_all_individuals[cli_stats[-1]].contributor_spops, pool_of_all_individuals[cli_stats[-1]].innovation_in_time, pool_of_all_individuals[cli_stats[-1]].avg_migration_jump]
                # self.update_stats(g, pool_of_all_individuals)
                if hof_savefile is not None:
                    self.current_generation = g
                    self.time_elapsed += time.process_time() - time0
                    self.save_state(get_state_filename(hof_savefile))
            if hof_savefile is not None:
                self.save_genotypes(hof_savefile)
                
            return self.hof, self.stats, df
    
    def add_to_worst(self):
        self.populations[0] = fill_population_with_random_frams(
            population=self.populations[0],
            dimensions=self.dimensions,
            evaluate=self.evaluate
        )
        for i in self.populations[0].population:
            if not hasattr(i, 'innovation_in_time'):
                i.innovation_in_time = [0.0 for _ in range(self.number_of_epochs)]
                i.innovation_in_time[self.current_epoch] = 1.0
                i.contributor_spops = [0.0 for _ in range(self.number_of_populations+1)]
                i.contributor_spops[-1] = 1.0
                i.prev_spop = 0

                # FIXME - same approach as in contributor_spops?
                i.avg_migration_jump = [0.0 for _ in range(self.number_of_populations*2 + 1)]
                i.avg_migration_jump[0] = 1.0
