from pandas import DataFrame
import time
import numpy as np
import copy
import json
from evolalg.json_encoders import Encoder
from evolalg.structures.population_methods import remove_excess_individuals_random, get_random_frams_solution
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
        ExperimentFrams.__init__(self, frams_lib=frams_lib, optimization_criteria=optimization_criteria,
                                 hof_size=hof_size, popsize=popsize,
                                 genformat=genformat, save_only_best=save_only_best, constraints=constraints)
        # super().__init__(popsize, hof_size, number_of_populations, migration_interval, save_only_best)
        self.number_of_epochs: int = None
        self.current_epoch: int = None
        self.results_directory_path = results_directory_path
        self.number_of_populations = number_of_populations
        self.migration_interval = migration_interval

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
            self.populations[pop_idx] = get_random_frams_solution(
                population=self.populations[pop_idx], evaluate=self.evaluate
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
                        # prev_spop_idx[self.current_epoch-1] = 1.0
                        prev_spop_idx[individual.prev_spop] = 1.0
                        individual.contributor_spops = list(
                            ((self.current_epoch-1) * np.array(individual.contributor_spops) + np.array(prev_spop_idx)) / self.current_epoch
                        )
                        # individual.avg_migration_jump = individual.avg_migration_jump + abs(individual.prev_spop - cur_spop)
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
            # self.update_stats(g, pool_of_all_individuals)
            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats, df

    def evaluate(self, genotype):
        data = self.frams_lib.evaluate([genotype])
        # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
        valid = True
        try:
            first_genotype_data = data[0]
            evaluation_data = first_genotype_data["evaluations"]
            default_evaluation_data = evaluation_data[""]
            fitness = [default_evaluation_data[crit] for crit in self.optimization_criteria][0]
        # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
        except (KeyError, TypeError) as e:
            valid = False
            print('Problem "%s" so could not evaluate genotype "%s", hence assigned it fitness: %s' % (
                str(e), genotype, BAD_FITNESS))
        if valid:
            default_evaluation_data['numgenocharacters'] = len(genotype)  # for consistent constraint checking below
            valid = self.check_valid_constraints(genotype, default_evaluation_data) 
        if not valid:
            fitness = BAD_FITNESS
        return fitness
        

    def mutate(self, gen1):
        return self.frams_lib.mutate([gen1])[0]

    def cross_over(self, gen1, gen2):
        return self.frams_lib.crossOver(gen1, gen2)
