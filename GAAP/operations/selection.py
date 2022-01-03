# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:20:28 2021

@author: kylei

"""
from numba import njit
import numpy as np

@njit
def random_selection(population, num_mating, tournament_size=None):
    """"Has O(num_mating) complexity"""

    indices = np.empty(1, dtype='int64')
    while len(indices) <= num_mating:  # Because indices starts at len == 1
        idx = np.random.randint(0, population.shape[0])
        indices = np.append(indices, idx)

    indices = np.delete(indices, 0)
    return indices


@njit(nogil=True)
def tournament_selection(fitnesses, num_mating, tournament_size, replacement):
    
    indices = np.empty(1, dtype='int64')
    while len(indices) <= num_mating:
        # First find a tournament of samples
        entries = np.random.randint(0, len(fitnesses),
                                    size=(tournament_size,))
        # Next get the fitnesses using the tournament
        tourney = fitnesses[entries]
        best_fit = np.argmax(tourney)
        best_sol = entries[best_fit]
        
        if not replacement:
            fitnesses = np.delete(fitnesses, best_sol)
        indices = np.append(indices, best_sol)

    indices = np.delete(indices, 0)
    return indices


@njit(nogil=True)
def roulette_wheel(fitnesses, num_mating, tournament_size=None, replacement=None):
    W = fitnesses[-1] # Fitnesses is is already presorted

    indices = np.empty(1, dtype='int64')
    while len(indices) <= num_mating:
        random_sample = np.random.randint(0, len(fitnesses))
        probability = fitnesses[random_sample]/W
        acceptance = np.random.uniform(0, 1)

        if acceptance < probability:
            indices = np.append(indices, random_sample)
        else:
            pass

    indices = np.delete(indices, 0)
    return indices


class Selector():
    def __init__(self, name, num_mating, tournament_size=None, replacement=False):
        if name == "stochastic_roulette":
            self.operator = roulette_wheel
        elif name == "tournament":
            self.operator = tournament_selection
            assert (not tournament_size == None)
        elif name == "random":
            self.operator = random_selection

        self.name = name
        self.num_mating = num_mating
        self.t_size = tournament_size
        self.replacement = replacement
        
        
    def operate(self, fitnesses):
        indices = self.operator(fitnesses,
                      num_mating = self.num_mating,
                      tournament_size=self.t_size,
                      replacement=self.replacement)
        
        return indices
    
