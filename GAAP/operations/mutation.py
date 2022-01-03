# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 20:24:51 2021

@author: kylei
"""

from numba import njit, guvectorize, float64, int64
import numpy as np


@njit(nogil=True)
def HDP_mutation(sol, UB, LB, prob, eta):
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            r = np.random.uniform(0, 1)
            
            if r <= prob:
                
                delt1 = ((sol[i, j] - LB[j]) / ((UB[j] - LB[j])))
                delt2 = (UB[j] - sol[i, j]) / (UB[j] - LB[j])
                
                r = np.random.uniform(0, 1)
                if r <= 0.5:
                    a = (2*r) + (1-2*r)
                    b = (1-delt1)**(eta+1)
                    c = (1/(eta+1))
                    deltk = ((a * b)**c)-1
                else:
                    a = 2*(1-r) + 2*(r-0.5)
                    b = (1-delt2)**(eta+1)
                    c = (1/(eta+1))
                    deltk = 1 - ((a * b)**c)
                
                final = sol[i, j] + (deltk*((UB[j] - LB[j])))
                sol[i, j] = final
            else:
                pass
            
@njit(nogil=True) 
def random_mutation(sol, UB, LB, prob):
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            r = np.random.uniform(0, 1)
            if r <= prob:
                sol[i, j] = np.random.randint(LB[j], UB[j])
            else:
                pass

class Mutator:
    def __init__(self, prob, eta=None):
        self.prob = prob
        self.eta = eta
        self.main_UB = []
        self.main_LB = []
        self.aux_UB = []
        self.aux_LB = []

    def operate(self, population, aux_population):
        HDP_mutation(population, self.main_UB, self.main_LB, self.prob, self.eta)
        random_mutation(aux_population, self.aux_UB, self.aux_LB, prob=self.prob)
        return None
    
        
    def SetBounds(self, mainUB, mainLB, auxUB, auxLB):
        self.main_UB = mainUB
        self.main_LB = mainLB
        self.aux_UB = auxUB
        self.aux_LB = auxLB