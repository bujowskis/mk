class Individual:
    def __init__(self, evaluate):
        self.genotype: str = ""  # not necessarily...
        self.rawfitness: float = None  # used for stats. This is raw fitness value, None = not evaluated or invalid genotype
        # float below may be misleading; set in case of multicriteria opt problems (not part of project though)
        self.fitness: float = None  # used in selection and can be modified e.g. by diversity or niching techniques
        self.evaluate = evaluate
    
    def copyFrom(self, individual):  # "copying constructor"
        self.genotype = individual.genotype
        self.rawfitness = individual.rawfitness
        self.fitness = individual.fitness
        return self

    def setAndEvaluate(self, genotype):
        self.genotype = genotype
        fitness = self.evaluate(genotype)  # re-using function frams_evaluate() from FramsticksEvolution.py, so tailoring second argument...
        if fitness is not None:  # BAD_FITNESS is None and indicates that the genotype was not valid or some other problem occurred during evaluation
            fitness = fitness[0]  # get first criterion
            if fitness < 0:
                fitness = 0
        self.fitness = self.rawfitness = fitness


    def __str__(self):
        try:
            return "%g\t%g\t'%s'" % (self.rawfitness, self.fitness, self.genotype)
        except:
            return "%g\t'%s'" % (self.rawfitness, self.genotype)

    def __gt__(self, other):
        if(self.rawfitness>other.rawfitness):
            return True
        else:
            return False