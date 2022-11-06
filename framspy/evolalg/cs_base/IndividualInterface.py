# from ..FramsticksLib.FramsticksLib import FramsticksLib

from __future__ import annotations


class IndividualInterface:
    genotype = None
    fitness = None

    def evaluate(self):
        pass

    def crossover(self, other: IndividualInterface):
        pass

    def mutate(self):
        pass

    def set_genotype(self, genotype):
        self.genotype = genotype
        # evaluate?
        pass

    def __str__(self):
        pass

    def __gt__(self, other):
        ...

    def __lt__(self, other):
        ...

    def __eq__(self, other):
        ...


class Individual:
    def __init__(self, evaluate):
        self.genotype: str = ""
        self.raw_fitness: float = None  # used for stats. This is raw fitness value, None = not evaluated or invalid genotype
        self.fitness: float = None  # used in selection and can be modified e.g. by diversity or niching techniques
        self.evaluate = evaluate

    def copy_from(self, individual):  # "copying constructor"
        self.genotype = individual.genotype
        self.raw_fitness = individual.raw_fitness
        self.fitness = individual.fitness
        return self

    def set_and_evaluate(self, genotype):
        self.genotype = genotype
        fitness = self.evaluate(genotype) # re-using function frams_evaluate() from FramsticksEvolution.py, so tailoring second argument...

        if fitness is not None:  # BAD_FITNESS is None and indicates that the genotype was not valid or some other problem occurred during evaluation
            fitness = fitness[0]  # get first criterion
            if fitness < 0:
                fitness = 0
        self.fitness = self.raw_fitness = fitness

    """ NOT SURE if it' needed to be coimplemented with FramsLib"""

    # def mutate(self):
    #     return self.frams_lib.mutate([self])[0]
    #
    # def cross_over(self, other):
    #     return self.frams_lib.crossOver(self, other)

    def __str__(self):
        try:
            return "%g\t%g\t'%s'" % (self.raw_fitness, self.fitness, self.genotype)
        except:
            return "%g\t'%s'" % (self.raw_fitness, self.genotype)

    def __gt__(self, other):
        if (self.raw_fitness > other.raw_fitness):
            return True
        else:
            return False