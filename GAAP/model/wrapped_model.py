# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 02:58:47 2021

@author: kylei
"""

import tensorflow as tf
import keras
import numpy as np
from . import keras_backend as kb
import copy
from numba import jit

class WrappedModel:
    def __init__(self, X, y, model, structure, functions, mode, fitness):
        """
        Necessary components to pass to GAAP to calculate fitness and
        rebuild neural network models.

        Parameters
        ----------
        X : numpy.ndarray
            Data inputs for calculating fitness.
        y : numpy.ndarray
            Data labels for calculating fitness.
        model : keras.model
            Contains a base model, necessary for getting maximum shape.
        structure : [str]
            Contains the structure of the model. Input, Dense, etc. 
        functions : [str]
            Contains user desired activation functions to optimize.
        mode : str
            Defines either one of two modes, ANN or CNN.
        fitness : str
            What fitness method to use (BCE, CCE, MSE).

        Returns
        -------
        None.

        """
        # User passed args
        self.data_inputs = X
        self.data_outputs = y
        self.model = model
        self.structure = structure
        self.functions = functions
        self.orig_weights = kb.unpackage_weights(model)  # May not need this
        self.mode = mode

        # Non-user args
        self.shapes = kb.get_layer_shapes(model)
        self.num_layers = structure.count("Dense")
        self.fitness_name=fitness

        # Gene boundaries
        self.main_UB = []
        self.main_LB = []
        self.aux_UB = []
        self.aux_LB = []

        # Gene count
        self.main_genes = 0
        self.aux_genes = 0
        
        if self.mode == "ANN":
            self.BoundCollector(model)
        elif self.mode == "CNN":
            self.BoundCollectorCNN(model)
            print("Bounds collected")
            
        self.SetFitness(fitness)

        self.test = None

    def SetFitness(self, fitness_func):
        """
        Allows variability of fitness calculation
        for different models/problems.

        Parameters
        ----------
        fitness_func : str
            Determines fitness method to use

        Returns
        -------
        tf.keras.losses.CategoricalCrossentropy()
            Occurs when desired fitness is CCE, used for 2< classification.
        tf.keras.losses.CategoricalCrossentropy()
            Occurs when desired fitness is BCE, used for binary classification.
        tf.keras.losses.mean_squared_error()
            Occurs when desired fitness is MSE, used for ANN regressors.

        """
        if fitness_func == "categorical_crossentropy":
            self.fitness = tf.keras.losses.CategoricalCrossentropy()
        elif fitness_func == "binary_crossentropy":
            self.fitness = tf.keras.losses.BinaryCrossentropy()
        elif fitness_func == "mean_squared_error":
            self.fitness = tf.keras.losses.MeanSquaredError()
        else:
            ValueError(fitness_func)
    
    def BoundCollectorCNN(self, model):
        # Get continuous bounds
        orig_weights = kb.unpackage_weights(model)
        weight_copy = np.array(copy.deepcopy(orig_weights))
        
        self.main_UB = weight_copy + np.random.uniform(-1., 1., (weight_copy.shape))
        self.main_LB = np.full(orig_weights.shape[0], np.min(orig_weights))
        
        aux_UB = []
        aux_LB = []
        for layer in model.layers[:-1]:
            if type(layer) == tf.python.keras.layers.convolutional.Conv2D:
                # Format is filter_count, filter_shape (same), filter_activation
                filter_count = layer.filters
                filter_shape = layer.kernel_size[0]+4
                filter_act = len(self.functions)
    
                aux_UB = np.append(aux_UB, [filter_count, filter_shape, filter_act])
                aux_LB = np.append(aux_LB, [1, 1, 0])
    
            elif (
                type(layer) == tf.python.keras.layers.pooling.MaxPooling2D
                or type(layer) == tf.python.keras.layers.pooling.AveragePooling2D
            ):
                # Format is pool_stride, pool_shape, pool_type
                pool_shape = layer.pool_size[0] + 2
                pool_type = len(["Max", "Mean"])
    
                aux_UB = np.append(aux_UB, [pool_type, pool_shape])
                aux_LB = np.append(aux_LB, [1, 1])
    
            elif type(layer) == tf.python.keras.layers.core.Dense:
                aux_UB = np.append(aux_UB, len(self.functions))
                aux_UB = np.append(aux_UB, layer.units)
    
                aux_LB = np.append(aux_LB, 0)  # Min function index
                aux_LB = np.append(aux_LB, layer.units / 2)  # Min number of units per layer
    
        self.aux_UB = aux_UB.astype("int32")
        self.aux_LB = aux_LB.astype("int32")
    
    def BoundCollector(self, model):
        """
        Prepare the gene boundaries based on the passed model and functions.

        Returns
        -------
        None.

        """
        orig_weights = kb.unpackage_weights(model)
        weight_copy = np.array(copy.deepcopy(orig_weights))
        self.main_UB = weight_copy + np.random.uniform(-1., 1., (weight_copy.shape))
        self.main_LB = np.full(orig_weights.shape[0], np.min(orig_weights))

        for layer in model.layers[:-1]:
            self.aux_UB.append(len(self.functions))
            self.aux_UB.append(layer.units)

            self.aux_LB.append(0)  # Min function index
            self.aux_LB.append(layer.units / 2)  # Min number of units per layer

        self.aux_UB = np.array(self.aux_UB, dtype="int32")
        self.aux_LB = np.array(self.aux_LB, dtype="int32")

        self.main_genes = len(orig_weights)
        # Input is never considered as a layer, and each layer has 2 genes
        self.aux_genes = self.structure.count("Dense")

    def GenerateAroundOrig(self, starting_solutions):
        # Generate main population uniformly
        main_population = np.empty(
            (1, self.orig_weights.shape[0]), dtype="float64"
        )
        weight = copy.deepcopy(self.orig_weights)
        new_solution = np.array(weight)
        
        main_population = np.vstack((main_population, new_solution))
        
        for i in range(starting_solutions-1):
            weight = copy.deepcopy(self.orig_weights)
            new_solution = np.array(weight) + np.random.uniform(-1, 1, (self.orig_weights.shape[0],))
                
            main_population = np.vstack((main_population, new_solution))
        
        # Delete empty value at axis 0 because vstack
        main_population = np.delete(main_population, 0, axis=0)
        
        # Collect original discrete values
        original_discrete = []
        for layer in self.model.layers[:-1]:
            activation = np.where(
                np.char.find(
                    self.functions, layer.activation.__name__
                             ) == 0)[0][0]
            units = layer.units
            original_discrete.append(activation)
            original_discrete.append(units)
            
        original_discrete = np.array(original_discrete, dtype="int32").reshape(1,-1)
        aux_population = np.empty((1, self.aux_genes*2), dtype="int32")
        
        for _ in range(starting_solutions):
            aux_population = np.vstack((aux_population, original_discrete))
        
        # Delete empty value at axis 1 because hstack
        aux_population = np.delete(aux_population, 0, axis=0)
        
        return (main_population, aux_population)
    
    def GenerateAroundOrigCNN(self, starting_solutions):
        # Generate main population uniformly
        main_population = np.empty(
            (1, self.orig_weights.shape[0]), dtype="float64"
        )
        weight = copy.deepcopy(self.orig_weights)
        new_solution = np.array(weight)
        
        main_population = np.vstack((main_population, new_solution))
        
        for i in range(starting_solutions-1):
            weight = copy.deepcopy(self.orig_weights)
            new_solution = np.array(weight) + np.random.uniform(
                -1, 1, (self.orig_weights.shape[0],)
            )
            
            main_population = np.vstack((main_population, new_solution))
        
        # Delete empty value at axis 0 because vstack
        main_population = np.delete(main_population, 0, axis=0)
        
        # Collect original discrete values
        
        aux_population = generate_aux(
            self.aux_LB, self.aux_UB, starting_solutions
            )
        
        return (main_population, aux_population)
        

    def Resample(self, worst, main_population, aux_population, amount,
                 fitness_past, fitness_present):
        
        n = int(main_population.shape[0]/2)
        
        # Take half of the lowest perfoming samples in the population
        indices_to_delete = worst[:n]
        
        best_main = main_population[worst[-1]]
        best_aux = aux_population[worst[-1]]
        
        fitness_change = (fitness_present - fitness_past)/fitness_past
        if fitness_change < .05: # Assuming if there was less than a 5% change
            bottom_optimal = main_population[worst[n]]
            self.main_LB -= best_main - bottom_optimal
            self.main_UB += best_main - bottom_optimal

        # Find the best solution
        main_population = np.delete(main_population, indices_to_delete, axis=0)
        aux_population = np.delete(aux_population, indices_to_delete, axis=0)
        
        for i in range(n):
            weight = copy.deepcopy(best_main)
            new_solution = np.array(weight) + np.random.uniform(
                self.main_LB, self.main_UB, (self.orig_weights.shape[0],)
            )
            
            main_population = np.vstack((main_population, new_solution))
            aux = copy.deepcopy(best_aux)
            np.vstack((aux_population, aux))
            
        """
        for i in range(n/2):
            # Make half of new solutions use best discrete parameters
            aux = copy.deepcopy(best_aux)
            np.vstack((aux_population, aux))
        """  
        
    def BuildBest(self, solution, aux_solution):
        # Convert (-1,) shape array to keras layer shape matrix
        # NOTE: shapes should be the same as the original network
        

        # Initialize a new Keras Sequential model
        model = keras.Sequential()

        # Rebuild the model using values
        if self.mode == "ANN":
            kb.rebuild_model(
                model, self.shapes, self.functions, self.structure, aux_solution
            )
        
        elif self.mode == "CNN":
            kb.rebuild_model_CNN(
                model, self.shapes, self.functions, self.structure, aux_solution
            )

        # Change the dimensions of the weights
        """
        changed_weights = kb.weight_dim_change(
            weights_as_matrix, aux_solution, self.num_layers
        )
        """
        
        # Recompile the model, should do nothing but establish graph
        model.compile(loss="categorical_crossentropy")
        model.fit(
            self.data_inputs, self.data_outputs, epochs=0, batch_size=100, verbose=0
        )
        
        new_shapes = kb.get_layer_shapes(model)
 
        changed_weights = kb.repackage_weights(solution, new_shapes)
        
        # Set the weights to the reduced version
        model.set_weights(changed_weights)


        return model

    def CalculateFitness(self, solution, aux_solution):
        """
        Rebuilds individual into Keras model and calculates CCE

        Parameters
        ----------
        solution : np.ndarray
            (-1,) shape ndarray with continuous genes.
        aux_solution : np.ndarray
            (-1,) shape ndarray with discrete genes.

        Returns
        -------
        solution_fitness : float64
            DESCRIPTION.

        """

        # Initialize a new Keras Sequential mdoel
        model = keras.Sequential()
        
        # Rebuild the model using values
        if self.mode == "ANN":
            kb.rebuild_model(
                model, self.shapes, self.functions, self.structure, aux_solution
            )
        elif self.mode == "CNN":
            kb.rebuild_model_CNN(
                model, self.shapes, self.functions, self.structure, aux_solution
                )
        else: 
            ValueError("Invalid mode: Rebuild model failed.")
            
        # Recompile the model, should do nothing but establish graph
        model.compile(loss=self.fitness)
        model.fit(self.data_inputs, self.data_outputs, epochs=0, batch_size=100)
        
        # Convert (-1,) shape array to keras layer shape matrix
        new_shapes = kb.get_layer_shapes(model)
 
        changed_weights = kb.repackage_weights(solution, new_shapes)
        
        # Set the weights to the reduced version
        model.set_weights(changed_weights)
        
        # Get predictions
        prediction = model(self.data_inputs)
        
        # Calculate fitness
        solution_fitness = 1.0 / (
            self.fitness(self.data_outputs, prediction).numpy() + 1e-2
        )

        if np.isnan(solution_fitness):
            solution_fitness = 1e-2

        return solution_fitness
    
def generate_aux(aux_LB, aux_UB, number):
    aux_population = np.empty((number, 1), dtype="int32")
    for lb, ub in zip(aux_LB, aux_UB):
        aux_gene = np.random.randint(lb, ub, (number,1))
        aux_population = np.hstack((aux_population, aux_gene))
        
    aux_population = np.delete(aux_population, 0, axis=1)
    return aux_population
    


def get_pred_labels(y_pred):
    pred = np.empty((1,), dtype="float64")
    for i in range(len(y_pred)):
        pred = np.append(pred, [np.argmax(y_pred[i])], axis=0)

    pred = np.delete(pred, 0, axis=0)

    return pred

"""
    def GeneratePops(self, starting_solutions):
        # Generate main population uniformly
        main_population = np.random.uniform(
            self.main_LB.min(),
            self.main_UB.max(),
            (starting_solutions, self.main_genes),
        )
        # Create empty nd.array to hold aux individuals
        aux_population = np.empty((starting_solutions, 1), dtype="int32")

        for i in range(self.aux_genes):
            act_column = np.random.randint(
                0, len(self.functions), size=(starting_solutions, 1)
            )
            units_column = np.random.randint(
                1, self.aux_UB[i], size=(starting_solutions, 1)
            )
            aux_population = np.hstack((aux_population, act_column, units_column))

        aux_population = np.delete(aux_population, 0, axis=1)

        return (main_population, aux_population)
"""
