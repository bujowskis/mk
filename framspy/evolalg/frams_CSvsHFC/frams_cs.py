import pandas as pd
import os
import time
import copy
import numpy as np
from evolalg.structures.population_methods import reinitialize_population_with_random_frams
from evolalg.cs_base.experiment_convection_selection_equiwidth import ExperimentConvectionSelectionEquiwidth
from evolalg.utils import get_state_filename
from ..frams_base.experiment_frams import ExperimentFrams

from evolalg.base.random_sequence_index import RandomIndexSequence
from evolalg.structures.individual import Individual
from evolalg.constants import BAD_FITNESS


class ExperimentFramsCSEquiwidth(ExperimentConvectionSelectionEquiwidth, ExperimentFrams):
    def __init__(self, frams_lib, optimization_criteria, hof_size,
                 popsize, constraints, genformat,
                 number_of_populations, migration_interval, save_only_best,
                 results_directory_path) -> None:
        ExperimentFrams.__init__(self, frams_lib=frams_lib, optimization_criteria=optimization_criteria,
                                hof_size=hof_size, popsize=popsize,
                                genformat=genformat, save_only_best=save_only_best, constraints=constraints)
        ExperimentConvectionSelectionEquiwidth.__init__(self, popsize=popsize, hof_size=hof_size, 
                                                number_of_populations=number_of_populations, 
                                                migration_interval=migration_interval, 
                                                save_only_best=save_only_best)
        self.number_of_epochs: int = None
        self.current_epoch: int = None
        self.results_directory_path = results_directory_path

    def evolve(
            self, hof_savefile, generations, initialgenotype, pmut, pxov, tournament_size,
            genformat,  # to enable in-code disabling of loading saved savefile
            constrains, repetition, migration_interval, number_of_populations, subpopsize, try_from_saved_file: bool = True
    ):
        initialgenotype = self.frams_getsimplest(genetic_format=genformat, initial_genotype=initialgenotype)
        self.setup_evolution(hof_savefile, initialgenotype, try_from_saved_file)
        self.number_of_epochs: int = int(np.floor(generations / self.migration_interval) + 1)  # account for epoch 0 (before start of migration)
        self.current_epoch: int = 0
        time0 = time.process_time()

        pool_of_all_individuals = []
        for p in self.populations:
            pool_of_all_individuals.extend(p.population)

        if self.current_generation == 0:  # if not continuing from hof, initialize all params
            for pop_idx in range(len(self.populations)):
                self.populations[pop_idx] = reinitialize_population_with_random_frams(
                    self, framslib=self.frams_lib, genformat=self.genformat,
                    population=self.populations[pop_idx], evaluate=self.evaluate,
                    constraints=self.constraints, initial_genotype=initialgenotype
                )
                for i in self.populations[pop_idx].population:
                    i.contributor_spops = [0.0 for _ in range(self.number_of_populations)]
                    i.avg_migration_jump = [0.0 for _ in range(self.number_of_populations*2 + 1)]

        df = pd.DataFrame(columns=['generation', 'total_popsize', 'best_fitness', 'contributor_spops', 'avg_migration_jump'])
        
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
                        prev_spop_idx = [0.0 for _ in range(self.number_of_populations)]
                        prev_spop_idx[individual.prev_spop] = 1.0
                        individual.contributor_spops = list(
                            ((self.current_epoch-1) * np.array(individual.contributor_spops) + np.array(prev_spop_idx)) / self.current_epoch
                        )
                        avg_mig = [0.0 for _ in range(self.number_of_populations*2 + 1)]
                        
                        no_change_idx = len(avg_mig) // 2
                        avg_mig[no_change_idx + cur_spop - individual.prev_spop] = 1.0

                        # avg_mig[individual.prev_spop - cur_spop] = 1.0
                        individual.avg_migration_jump = list(
                            ((self.current_epoch-1) * np.array(individual.avg_migration_jump) + np.array(avg_mig)) / self.current_epoch
                        )

            pool_of_all_individuals = []
            for p in self.populations:
                pool_of_all_individuals.extend(p.population)
            self.update_stats(g, pool_of_all_individuals)
            cli_stats = self.get_cli_stats()
            df.loc[len(df)] = [cli_stats[0], cli_stats[1], cli_stats[2], pool_of_all_individuals[cli_stats[-1]].contributor_spops, pool_of_all_individuals[cli_stats[-1]].avg_migration_jump]
            
            file_path = f'results/frams/cs/frams_CSvsHFC_cs-{genformat}-{constrains["max_numjoints"]}-{constrains["max_numconnections"]}-{constrains["max_numgenochars"]}-{constrains["max_numneurons"]}-{repetition}-{migration_interval}-{number_of_populations}-{subpopsize}-{pmut}-{pxov}-{tournament_size}.csv'
            file_exists = os.path.exists(file_path)
            if file_exists:
                df_existing = pd.read_csv(file_path)
                df = pd.concat([df_existing, df])
                df = df.drop_duplicates(subset=['generation'], keep='first')
            df.to_csv(file_path, mode='w', index=False, header=['generation', 'total_popsize', 'best_fitness', 'contributor_spops', 'innovation_in_time', 'avg_migration_jump'])

            if hof_savefile is not None:
                self.current_generation = g
                self.time_elapsed += time.process_time() - time0
                self.save_state(get_state_filename(hof_savefile))
        if hof_savefile is not None:
            self.save_genotypes(hof_savefile)

        return self.hof, self.stats, df
    
    def evaluate(self, genotype):
        return super().evaluate(genotype)
    

    def make_new_population(self, individuals, prob_mut, prob_xov, tournament_size):  # fixme - rename to evolve_one_step
        """'individuals' is the input population (a list of individuals).
        Assumptions: all genotypes in 'individuals' are valid and evaluated (have fitness set).
        Returns: a new population of the same size as 'individuals' with prob_mut mutants, prob_xov offspring, and the remainder of clones."""

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
                new_individual.contributor_spops = copy.deepcopy(ind.contributor_spops)
                new_individual.avg_migration_jump = copy.deepcopy(ind.avg_migration_jump)
                newpop.append(new_individual)

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind2 = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            new_individual = Individual()
            new_individual.set_and_evaluate(self.cross_over(ind1.genotype, ind2.genotype), self.evaluate)
            if new_individual.fitness is not BAD_FITNESS:
                new_individual.contributor_spops = list(np.average([ind1.contributor_spops, ind2.contributor_spops], axis=0))
                new_individual.avg_migration_jump = list(np.average([ind1.avg_migration_jump, ind2.avg_migration_jump], axis=0))
                newpop.append(new_individual)

        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < self.popsize:
            ind = self.select(individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind_copy = Individual().copyFrom(ind)
            ind_copy.contributor_spops = copy.deepcopy(ind.contributor_spops)
            ind_copy.avg_migration_jump = copy.deepcopy(ind.avg_migration_jump)
            newpop.append(ind_copy)

        return newpop