from abc import ABC, abstractmethod

from evolalg.structures.population import PopulationStructures
from evolalg.structures.individual import Individual

"""
ABC specification of migration for truly modular approach

NOTE - problem with overriding the way `evolve` works - problems where to place the code
"""
