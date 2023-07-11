# fixme - make into a class, handle RIS by itself (!!)
#   also: RIS introduce select_tournament

# todo (maybe) - different selection methods could use a "universal parser" (kwargs, args)
#   the need may arise when implementing FUSS (or other selection methods)
#   instead of "universal parser" -> just accept any arguments, make methods look for their needed arguments

def select_tournament(individuals, tournament_size, random_index_sequence):
    """Tournament selection, returns the index of the best individual from those taking part in the tournament"""
    best_index = None
    for i in range(tournament_size):
        rnd_index = random_index_sequence.getNext()
        if best_index is None or individuals[rnd_index].fitness > best_index.fitness:
            best_index = individuals[rnd_index]
    return best_index


# class TournamentSelection:
#     ris: RIS
