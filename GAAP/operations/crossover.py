# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:24:41 2021

@author: kylei
"""
from numba import njit
import numpy as np

@njit(nogil=True)
def single_point_crossover(population, indices, prob=0.9, eta=None):
    prob = 1
    children = np.empty((1, population.shape[1]), dtype='int32')
    iterations = int(len(indices)/2)  # Because two mates to a pair
    
    indx = 0
    for i in range(iterations):
        if np.random.binomial(1, prob) == 1:
            parent1 = population[indices[indx]]
            indx += 1
            parent2 = population[indices[indx]]
            indx += 1
    
            point = np.random.randint(0, parent1.shape[0])
    
            child1 = np.empty(parent1.shape, dtype="int32")
            child1[:point] = parent1[:point]
            child1[point:] = parent2[point:]
            
            child2 = np.empty(parent1.shape, dtype="int32")
            child2[:point] = parent2[:point]
            child2[point:] = parent1[point:]
            
            child1 = child1.reshape(1, -1)
            child2 = child2.reshape(1, -1)
    
            children = np.append(children, child1, axis=0)
            children = np.append(children, child2, axis=0)
        else:
            indx += 2
            
    #children = np.delete(children, 0) #Don't know why Numba hates this
    return children


@njit
def uniform_crossover(population, indices, prob=0.9, eta=None):
    children = np.empty((1, population.shape[1]), dtype='float64')
    iterations = int(len(indices)/2)  # Because two mates to a pair

    indx = 0
    for i in range(iterations):
        if np.random.binomial(1, prob) == 1:
            crossover_mask = np.random.randint(
                0, 2, size=(population.shape[1],))
            parent1 = population[indices[indx]]
            indx += 1
            parent2 = population[indices[indx]]
            indx += 1

            child1 = np.empty(parent1.shape, dtype="float64")
            child1[crossover_mask == 1] = parent1[crossover_mask == 1]
            child1[crossover_mask == 0] = parent2[crossover_mask == 0]

            child2 = np.empty(parent1.shape, dtype="float64")
            child2[crossover_mask == 1] = parent2[crossover_mask == 1]
            child2[crossover_mask == 0] = parent1[crossover_mask == 0]

            child1 = child1.reshape(1, -1)
            child2 = child2.reshape(1, -1)

            children = np.append(children, child1, axis=0)
            children = np.append(children, child2, axis=0)
        else:
            indx += 2
    return children


@njit(nogil=True)
def simulated_binary_crossover(population, indices, prob=0.9, eta=15):
    prob = 1
    children = np.empty((1, population.shape[1]), dtype='float64')
    iterations = int(len(indices)/2)

    indx = 0
    for i in range(iterations):
        if np.random.binomial(1, prob) == 1:
            parent1 = population[indices[indx]]
            indx += 1
            parent2 = population[indices[indx]]
            indx += 1

            u = np.random.uniform(0, 1)
            Bq = 0.
            if u <= 0.5:
                Bq = pow(2*u, 1/(eta+1))
            else:
                Bq = pow(1/(2*(1-u)), 1/(eta+1))

            child1 = 0.5*((1+Bq)*parent1 + (1-Bq)*parent2)
            child2 = 0.5*((1-Bq)*parent1 + (1+Bq)*parent2)

            child1 = child1.reshape(1, -1)
            child2 = child2.reshape(1, -1)
            
            children = np.append(children, child1, axis=0)
            children = np.append(children, child2, axis=0)
        else:
            indx += 2  # Skip the first set of parents
    return children

class Crossover():
    def __init__(self, name, prob, eta = None):
        if name == "single":
            self.operator = single_point_crossover
        elif name == "uniform":
            self.operator = uniform_crossover
        elif name == "SBX":
            self.operator = simulated_binary_crossover
            assert(not eta == None)
        self.name = name
        self.prob = prob
        self.eta = eta
        
    def operate(self, population, indices):
        children = self.operator(population, indices, self.prob, self.eta)
        children = np.delete(children, 0, axis=0)
          
        return children