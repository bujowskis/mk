from pandas import DataFrame
import time
import numpy as np
from evolalg.structures.population_methods import reinitialize_population_with_random_frams
from evolalg.cs_base.experiment_convection_selection_equiwidth import ExperimentConvectionSelectionEquiwidth
from evolalg.utils import get_state_filename
from ..frams_base.experiment_frams import ExperimentFrams


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
        self.number_of_epochs: int = int(np.floor(generations / self.migration_interval) + 1)  # account for epoch 0 (before start of migration)
        self.current_epoch: int = 0
        time0 = time.process_time()

        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)

        for pop_idx in range(len(self.populations)):
            self.populations[pop_idx] = reinitialize_population_with_random_frams(
                framslib=self.frams_lib, genformat=self.genformat, 
                population=self.populations[pop_idx], evaluate=ExperimentFrams.evaluate,
                initial_genotype=initialgenotype
                )
            for i in self.populations[pop_idx].population:
                i.contributor_spops = [0.0 for _ in range(self.number_of_populations)]
                i.avg_migration_jump = [0.0 for _ in range(self.number_of_populations*2 + 1)]

        df = DataFrame(columns=['generation', 'total_popsize', 'best_fitness', 'contributor_spops', 'avg_migration_jump'])

        for g in range(self.current_generation, generations):
            for p in self.populations:
                p.population = self.make_new_population(p.population, pmut, pxov, tournament_size)

            if g % self.migration_interval == 0:
                self.current_epoch += 1
                for idx, population in enumerate(self.populations):
                    for individual in population.population:
                        individual.prev_spop = idx
                # todo - contributor_spops here?
                self.migrate_populations()
                # todo - contributor_spops here?
                for cur_spop, population in enumerate(self.populations):
                    for individual in population.population:
                        prev_spop_idx = [0.0 for _ in range(self.number_of_populations)]
                        prev_spop_idx[individual.prev_spop] = 1.0
                        individual.contributor_spops = list(
                            ((self.current_epoch-1) * np.array(individual.contributor_spops) + np.array(prev_spop_idx)) / self.current_epoch
                        )
                        avg_mig = [0.0 for _ in range(self.number_of_populations*2 + 1)]
                        avg_mig[abs(individual.prev_spop - cur_spop)] = 1.0
                        individual.avg_migration_jump = list(
                            ((self.current_epoch-1) * np.array(individual.avg_migration_jump) + np.array(avg_mig)) / self.current_epoch
                        )

            pool_of_all_individuals = []
            for p in self.populations:
                pool_of_all_individuals.extend(p.population)
            self.update_stats(g, pool_of_all_individuals)
            cli_stats = self.get_cli_stats()
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2], pool_of_all_individuals[cli_stats[-1]].contributor_spops, pool_of_all_individuals[cli_stats[-1]].avg_migration_jump]
            
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats, df