import numpy as np

from ..base.modular_experiment_abc import ModularExperimentABC
from ..structures.hall_of_fame import HallOfFame
from ..structures.individual import Individual

from ..base.selection.tournament_selection import tournament_selection


class ExperimentNumerical(ModularExperimentABC):
    def __init__(self, hof_size, popsize, save_only_best) -> None:
        ModularExperimentABC.__init__(self, popsize=popsize, hof_size=hof_size, save_only_best=save_only_best)
        self.selection = TournamentSelection()  # TODO - THIS

    def mutate(self, gen1):
        return list(gen1 + np.random.randint(-10, 10, len(gen1)))

    def cross_over(self, gen1, gen2):
        return gen1

    def evaluate(self, genotype):
        return 1/sum([x*x for x in genotype])

    def select(self, individuals) -> Individual:
        return tournament_selection(
            individuals,
            tournament_size=self.selection.tournament_size,
            random_index_sequence=self.selection.random_index_sequence
        )

    def make_new_population(self, individuals, prob_mut, prob_xov, tournament_size):
        """'individuals' is the input population (a list of individuals).
        Assumptions: all genotypes in 'individuals' are valid and evaluated (have fitness set).
        Returns: a new population of the same size as 'individuals' with prob_mut mutants, prob_xov offspring, and the remainder of clones."""

        newpop = []
        N = len(individuals)
        expected_mut = int(N * prob_mut)
        expected_xov = int(N * prob_xov)
        assert expected_mut + \
               expected_xov <= N, "If probabilities of mutation (%g) and crossover (%g) added together exceed 1.0, then the population would grow every generation..." % (
            prob_mut, prob_xov)
        ris = RandomIndexSequence(N)

        # adding valid mutants of selected individuals...
        while len(newpop) < expected_mut:
            ind = self.select(
                individuals, tournament_size=tournament_size, random_index_sequence=ris)
            self.addGenotypeIfValid(newpop, self.mutate(ind.genotype))

        # adding valid crossovers of selected individuals...
        while len(newpop) < expected_mut + expected_xov:
            ind1 = self.select(
                individuals, tournament_size=tournament_size, random_index_sequence=ris)
            ind2 = self.select(
                individuals, tournament_size=tournament_size, random_index_sequence=ris)
            self.addGenotypeIfValid(
                newpop, self.cross_over(ind1.genotype, ind2.genotype))

        # select clones to fill up the new population until we reach the same size as the input population
        while len(newpop) < len(individuals):
            ind = self.select(
                individuals, tournament_size=tournament_size, random_index_sequence=ris)
            newpop.append(Individual().copyFrom(ind))

        return newpop
