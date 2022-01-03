# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 22:57:49 2021

@author: kylei
"""

import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib as mpl
import copy

# plt.style.use("science")
mpl.rcParams["figure.dpi"] = 300
plt.style.use("ieee")

class GAAP:
    def __init__(
        self,
        *,
        termination,
        selection,
        main_crossover,
        aux_crossover,
        mutation,
        wrapped_model,
    ):
        self.termination = termination
        # Operation related
        self.selection = selection
        self.main_crossover = main_crossover
        self.aux_crossover = aux_crossover
        self.mutation = mutation
        self.wrapped_model = wrapped_model
        self.mutation.SetBounds(
            self.wrapped_model.main_UB,
            self.wrapped_model.main_LB,
            self.wrapped_model.aux_UB,
            self.wrapped_model.aux_LB,
        )
        # Population related
        self.population = []
        self.aux_population = []
        self.fitness = []

        """Testing information"""
        self.fitness_gen_ = []  # To store best fitness for each generation
        self.resample_gen_ = []
        self.time = 0

    def optimize(
        self,
        starting_size,
        replacement=False,
        seed=192,
        save_gen=[],
        save_name=None,
        resample_gen=[],
        resample_amount=1,
        hover_gens=100
    ):

        np.random.seed(seed)
        
        # For resampling
        resampling_fitness_past = 1
        resampling_fitness_present = 0
        
        # Intitialize population
        if self.wrapped_model.mode == "ANN":
            pops = self.wrapped_model.GenerateAroundOrig(starting_size)
        elif self.wrapped_model.mode == "CNN":
            pops = self.wrapped_model.GenerateAroundOrigCNN(starting_size)

        # Not sure why this doesn't return both pops normally
        self.population = pops[0]
        self.aux_population = pops[1]

        start_time = time.time()
        cur_gen = 0
        number_to_delete = int(self.selection.num_mating / 2)
        while cur_gen < self.termination:
            self.fitness = list(
                map(
                    self.wrapped_model.CalculateFitness,
                    self.population,
                    self.aux_population,
                )
            )
            
            
            self.fitness = np.array(self.fitness, dtype="float64")
            # Find best and worst samples
            best = np.amax(self.fitness)
            self.fitness_gen_.append(best)
            worst = np.argsort(self.fitness, kind="heapsort")

            if cur_gen in resample_gen:
                resampling_fitness_present = best
                self.wrapped_model.Resample(
                    worst, self.population, self.aux_population, resample_amount,
                    resampling_fitness_past, resampling_fitness_present
                )
                self.resample_gen_.append([cur_gen, best])
                resampling_fitness_past = best
                
                self.mutation.SetBounds(self.wrapped_model.main_UB,
                                    self.wrapped_model.main_LB,
                                    self.wrapped_model.aux_UB,
                                    self.wrapped_model.aux_LB)
            else:
                worst = worst[:number_to_delete]
                # Delete all relevant information about worst samples
                self.population = np.delete(self.population, worst, axis=0)
                self.aux_population = np.delete(self.aux_population, worst, axis=0)
                self.fitness = np.delete(self.fitness, worst, axis=0)
                
                # Try to keep complexity low by removing some solutions
                if self.population.shape[0] > hover_gens:
                    number_to_delete = self.selection.num_mating * 2
                else:
                    number_to_delete = int(self.selection.num_mating / 2)

            if cur_gen in save_gen:
                self.SaveBestInner(cur_gen, save_name)

            # Verbose
            print("Generation: " + str(cur_gen))
            print("Fitness = %.6f" % best)
            print("Population = " + str(self.population.shape[0]))
            print("-"*10)
            print("Fitness len: " + str(len(self.fitness)))
            print("Main len: " + str(self.population.shape[0]))
            print("Aux len: " + str(self.aux_population.shape[0]) + "\n")

            # Select samples
            fit_copy = copy.deepcopy(self.fitness)
            indices = self.selection.operate(fit_copy)

            # Perform crossover
            main_children = self.main_crossover.operate(self.population, indices)
            aux_children = self.aux_crossover.operate(self.aux_population, indices)

            # Replace parents with children
            if replacement:
                self.population = np.delete(self.population, indices, axis=0)
                self.aux_population = np.delete(self.aux_population, indices, axis=0)

            # Append children to population
            self.population = np.append(self.population, main_children, axis=0)
            self.aux_population = np.append(self.aux_population, aux_children, axis=0)

            # Perform mutation
            
            self.mutation.operate(self.population, self.aux_population)

            cur_gen += 1
            np.random.seed(seed + cur_gen)
        self.time = time.time() - start_time
        return None

    def GetBestInner(self, cur_gen, save_name):
        """For saving a model at indicated generations"""
        best = np.amax(self.fitness)

        best_main = self.population[best]
        best_aux = self.aux_population[best]

        model = self.wrapped_model.BuildBest(best_main, best_aux)

        model.save(f"{save_name}-{cur_gen}")

    def GetBest(self, num_solutions=1):
        self.fitness = list(
            map(
                self.wrapped_model.CalculateFitness,
                self.population,
                self.aux_population,
            )
        )

        best = np.argsort(self.fitness, kind="heapsort")[::-1][:num_solutions]

        model_list = []
        best_mains = []
        best_auxs = []
        for idx in best:
            best_main = self.population[idx]
            best_aux = self.aux_population[idx]
            best_mains.append(best_main)
            best_auxs.append(best_auxs)

            model = self.wrapped_model.BuildBest(best_main, best_aux)
            model_list.append(model)

        return model_list, (best_main, best_aux)

    def troubleshoot_sizes(self, fitness):
        """Only used to troubleshoot populations and or fitness desyncing.

        Args:
            fitness ([float64]): Array containing fitness values of individuals 
        """
        print("Population before deletion: " + str(self.population.shape))
        print("Aux Population before deletion: " + str(self.aux_population.shape))
        print("Fitness before deletion: " + str(len(fitness)))
        print("\n")

        print("Population after deletion: " + str(self.population.shape))
        print("Aux Population after deletion: " + str(self.aux_population.shape))
        print("Fitness after deletion: " + str(len(fitness)))
        print("\n")

        print("-" * 60)

    def plot_time_gen(self, title="GAAP"):

        plt.plot(
            np.arange(0, self.termination),
            self.fitness_gen_,
            label="%.2f sec" % self.time,
        )
        if len(self.resample_gen_) != 0:
            self.resample_gen_ = np.array(self.resample_gen_)
            plt.scatter(
                self.resample_gen_[:, [0]],
                self.resample_gen_[:, [1]],
                label="Resampling",
                marker="o",
                color="red",
            )
        plt.legend(loc="upper left")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(title)
        plt.show()
