import numpy as np


class RandomIndexSequence:
    """A helper function for tournament selection. Allows each individual to take part in a tournament,
    thus reducing the risk of overlooking individuals by not sampling them due to uncontrolled
    randomness, as can happen in the naive implementation of tournament selection."""

    def __init__(self, popsize):
        self.popsize: int = popsize
        self.permut: list = []

    def getNext(self):
        if len(self.permut) < 1:
            self.permut = list(np.random.permutation(self.popsize))
        return self.permut.pop()
