# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:28:09 2021

@author: kylei
"""
#%%
import pandas as pd
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_blobs
import sys

sys.path.append("../GAAP")

from gaap.operations import Selector
from gaap.operations import Crossover
from gaap.operations import Mutator
from gaap.optimizer import GAAP
from gaap.model import WrappedModel

mpl.rcParams["figure.dpi"] = 300
plt.style.use("ieee")

n_classes = 3
data, labels = make_blobs(n_samples=5000,
                          centers=n_classes,
                          random_state=10,
                          center_box=(-40, 1),
                          cluster_std=2.5)

toy_dataset = pd.DataFrame(data = data)
toy_dataset[2] = labels


# Initial plotting of toy dataset
fig, ax = plt.subplots()

colours = []
for i in range(n_classes):
    colours.append([np.random.uniform(0,1),
                    np.random.uniform(0,1),
                    np.random.uniform(0,1)])
colours = ["tab:orange", "tab:blue", "tab:grey"]
for label in range(n_classes):
    ax.scatter(x=data[labels==label, 0], 
               y=data[labels==label, 1], 
               c=colours[label], 
               s=10, 
               label='Class {c}'.format(c = label))

ax.set(xlabel='X',
       ylabel='Y',
       title='Toy Dataset')

ax.legend(loc='upper right')
plt.show()

labels = tf.keras.utils.to_categorical(labels, n_classes)
#%%
if __name__ == "__main__":

    model = keras.Sequential()
    model.add(keras.Input(2,))
    model.add(keras.layers.Dense(20, activation="sigmoid", use_bias=False))
    model.add(keras.layers.Dense(20, activation="sigmoid", use_bias=False))
    model.add(keras.layers.Dense(20, activation="sigmoid", use_bias=False))
    model.add(keras.layers.Dense(3, activation="softmax", use_bias=False))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-1)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(data, labels, epochs=1, batch_size=1000)

    model.save("./toy_model")

    model = keras.models.load_model("toy_model")
    
    tf.keras.utils.plot_model(
        model, to_file='toy_model.png', show_shapes=True,
        show_layer_names=False, rankdir='LR', expand_nested=False, dpi=300,
    )
#%%
    structure = ["Input", "Dense", "Dense", "Dense", "OutputC"]
    funcs = [
        "relu",
        "sigmoid",
        "softmax",
        "softplus",
        "elu",
        "selu",
    ]

    wm = WrappedModel(
        data, labels, model, structure, funcs, "ANN", "categorical_crossentropy"
    )

    selection = Selector("stochastic_roulette",
                         num_mating=25,
                         tournament_size=20,
                         replacement=False)
    main_crossover = Crossover("SBX", prob=1, eta=5)
    aux_crossover = Crossover("single", prob=1)
    mutation = Mutator(prob=0.003, eta=5)

    GA_obj = GAAP(
        termination=100,
        selection=selection,
        main_crossover=main_crossover,
        aux_crossover=aux_crossover,
        mutation=mutation,
        wrapped_model=wm,
    )
    GA_obj.optimize(starting_size=50, replacement=False, seed=192,
                    resample_gen=[])
    #%%
    import matplotlib.pyplot as plt
    plt.plot(
            np.arange(0, len(GA_obj.fitness_gen_)),
            GA_obj.fitness_gen_,
            label="%.2f sec" % GA_obj.time,
        )
    if len(GA_obj.resample_gen_) != 0:
        GA_obj.resample_gen_ = np.array(GA_obj.resample_gen_)
        plt.scatter(
            GA_obj.resample_gen_[:, [0]],
            GA_obj.resample_gen_[:, [1]],
            label="Resampling",
            marker="o",
            color="red",
            s=8
        )
    plt.legend(loc="upper left")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("GAAP - Toy Dataset")
    plt.show()
    
#%%
    model, sol = GA_obj.GetBest()
    
    y_pred = model[0].predict(data)
#%%
    from sklearn.metrics import accuracy_score
    def OHtoList(arr):
        new_arr = []
        for i in range(arr.shape[0]):
            new_arr.append(np.argmax(arr[i]))
            
        new_arr = np.array(new_arr)
        return new_arr
    
    y_pred_s = OHtoList(y_pred)
    labels_s = OHtoList(labels)
    
    print("accuracy:" + str(accuracy_score(labels_s, y_pred_s)))

    activations = ["N/A"] # Accounts for initial input layer
    units = []
    for i in range(len(model.layers)):
        units.append(model.layers[i].units)
        activations.append(model.layers[i].activation.__name__)
        
#%%
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    
    cm = confusion_matrix(labels_s, y_pred_s)
    ConfusionMatrixDisplay(
        cm
        ).plot(cmap="gist_yarg")
    plt.title("GAAP - Toy Dataset - CM")
    plt.show()
    
    