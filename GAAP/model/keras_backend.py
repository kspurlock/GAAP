# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:51:33 2021

@author: kylei
"""
import numpy as np
from keras import layers


def unpackage_weights(model):
    model_weights = model.get_weights()

    ret_weights = np.empty((1,), float)

    for i in range(len(model_weights)):
        layer_weight = model_weights[i].ravel()
        ret_weights = np.append(ret_weights, layer_weight, axis=0)

    ret_weights = np.delete(ret_weights, 0, axis=0)

    return ret_weights


def repackage_weights(unpack_weights, shape_list):
    matrix_weights = []
    current_idx = 0

    for i in shape_list:  # Kind of sucks because non-linear
        acc = 1  # Accumulator
        for j in i:
            acc = acc * j

        selected_weights = unpack_weights[current_idx : (current_idx + acc)]
        selected_weights = np.array(selected_weights).reshape(i)
        matrix_weights.append(selected_weights)

        current_idx += acc

    return matrix_weights


def get_layer_shapes(model):
    model_weights = model.get_weights()

    shapes = []

    for i in range(len(model_weights)):
        layer_weight = model_weights[i]
        shapes.append(layer_weight.shape)

    return shapes


def weight_dim_change_BIAS(packaged_weights, aux_genes, num_layers):
    """
        NOTE: Not currently implemented. Not worth it to contain bias weights
              in main population
    """
    # Accounts for only the trainable layers
    # the output and input are not considered
    layer_indices = np.arange(1, num_layers * 2, step=2)

    # Requires specified number of layers (since each layer will have units
    # and activation)
    shape_indices = np.arange(num_layers - 1, aux_genes.shape[0])

    for i, j in zip(layer_indices, shape_indices):
        packaged_weights[i - 1] = packaged_weights[i - 1][:, : aux_genes[j]]
        packaged_weights[i] = packaged_weights[i][: aux_genes[j]]
        packaged_weights[i + 1] = packaged_weights[i + 1][: aux_genes[j], :]

    return packaged_weights  # Packaged_weights passed by reference


def weight_dim_change(packaged_weights, aux_genes, num_layers):
    # Which layers need to be modified from the packaged perspective
    layer_indices = np.arange(1, num_layers + 1)

    # Which genes need to be accessed
    # Need to access odd numbered genes for shapes
    gene_indices = np.arange(1, num_layers * 2, step=2)

    for layer, gene in zip(layer_indices, gene_indices):
        packaged_weights[layer - 1] = packaged_weights[layer - 1][:,:aux_genes[gene]]
        packaged_weights[layer] = packaged_weights[layer][:aux_genes[gene], :]
        
    return packaged_weights  # Packaged_weights passed by reference


def rebuild_model(model, shapes, funcs, structure, aux_genes):
    """
    Parameters
    ----------
    model : keras.Sequential
        An empty keras model.
    structure : list
        List of string values that describe the original model architecture.
    aux_genes : TYPE
        Genes that detail changes to model architecture.

    Returns
    -------
    None.

    """
    shape_counter = 1
    act_counter = 0
    for layer_idx in range(len(structure)):
        if structure[layer_idx] == "Dense":
            model.add(
                layers.Dense(
                    aux_genes[shape_counter],
                    activation=funcs[aux_genes[act_counter]],
                    use_bias=False,
                    kernel_initializer="normal",
                )
            )
            shape_counter += 2  # Oddly distributed in aux_population
            act_counter += 2

        # Used for classification models
        elif structure[layer_idx] == "OutputC":
            model.add(
                layers.Dense(shapes[-1][-1], activation="softmax", use_bias=False)
            )
        # Used for regressor models
        elif structure[layer_idx] == "OutputR":
            model.add(layers.Dense(shapes[-1][-1], activation="linear", use_bias=False))
            
        elif structure[layer_idx] == "Input":
            model.add(layers.Input((shapes[0][0],)))
        else:
            ValueError(structure[layer_idx])
    return None


def rebuild_model_CNN(model, shapes, funcs, structure, aux_genes):
        """
        Parameters
        ----------
        model : keras.Sequential
            An empty keras model.
        structure : list
            List of string values that describe the original model architecture.
        aux_genes : TYPE
            Genes that detail changes to model architecture.
    
        Returns
        -------
        None.
    
        """

        for layer_idx in range(len(structure)):
            if structure[layer_idx] == "Conv":
                model.add(
                    layers.Conv2D(
                        aux_genes[0],
                        (aux_genes[1], aux_genes[1]),
                        activation=funcs[aux_genes[2]],
                        use_bias=False,
                        kernel_initializer="normal",
                    )
                )
            elif structure[layer_idx] == "Pool":
                if aux_genes[3] == 1:  # MaxPooling
                    model.add(
                        layers.MaxPooling2D(
                            (aux_genes[4], aux_genes[4]),
                            strides=(3,3)
                        )
                    )
                elif aux_genes[3] == 2:  # AveragePooling
                    model.add(
                        layers.AveragePooling2D(
                            (aux_genes[4], aux_genes[4]),
                            strides=(3,3)
                        )
                    )
                else:
                    ValueError("Invalid Pool")

            elif structure[layer_idx] == "Flatten":
                model.add(layers.Flatten())

            elif structure[layer_idx] == "Dense":
                model.add(
                    layers.Dense(
                        aux_genes[6], activation=funcs[aux_genes[5]], use_bias=False
                    )
                )

            # Used for classification models
            elif structure[layer_idx] == "OutputC":
                model.add(
                    layers.Dense(shapes[-1][-1], activation="softmax", use_bias=False)
                )
            elif structure[layer_idx] == "Input":
                model.add(layers.Input((28, 28, 1))) # Currently set up for MNIST dataset
            else:
                ValueError(structure[layer_idx])
        return None
