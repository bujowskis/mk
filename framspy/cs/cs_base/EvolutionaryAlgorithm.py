import Population


class EvolutionaryAlgorithm:
    """
    Interface implemented by single-population Evolutionary Algorithms
    """
    population: Population
    current_generation: int = 0

    def __init__(self, population: Population):
        """
        :param population: initial population
        """
        self.population = population  # TODO - take as parameter OR initialize separately

    def evaluate_individuals(self):
        ...

    def select_for_mutation(self):
        ...

    def mutate_individuals(self):
        ...

    def select_for_crossover(self):
        ...

    def crossover_individuals(self):
        ...

    def evolve_step(self):
        """
        Makes one evolutionary step; i.e. evolve with generations == 1
        """
        ...

    def evolve(self, generations: int = None, stopping_condition=None):
        """
        Starts the evolution process for a fixed number of generations or until stopping condition is met
        :param generations: number of generations
        :param stopping_condition: condition that when met, stops the evolution
        """
        ...
