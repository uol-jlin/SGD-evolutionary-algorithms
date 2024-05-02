import os
import random
import time
from deap import base, creator, tools, algorithms
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
import logging

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if the dataset is already downloaded and load it
dataset_path = 'fashion_mnist.npz'
if os.path.exists(dataset_path):
    logging.info("Loading dataset from cache.")
    with np.load(dataset_path, allow_pickle=True) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
else:
    logging.info("Downloading dataset.")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    np.savez(dataset_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(hyperparameters):
    """
    Create and compile a neural network model based on specified SGD hyperparameters including L2 regularization and learning rate decay.
    """
    learning_rate, momentum, l2_reg, decay = hyperparameters
    logging.debug(f"Configuring model: LR={learning_rate}, Momentum={momentum}, L2={l2_reg}, Decay={decay}")
    model = Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Dense(10)
    ])
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def eval_model(hyperparameters):
    """
    Train the model and evaluate its performance on the validation dataset.
    """
    model = create_model(hyperparameters)
    logging.info(f"Training model with hyperparameters: {hyperparameters}")
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, verbose=0, validation_split=0.1)
    elapsed_time = time.time() - start_time
    val_accuracy = history.history['val_accuracy'][-1]
    logging.info(f"Training completed in {elapsed_time:.2f} seconds. Validation accuracy: {val_accuracy}")
    return (-val_accuracy,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float_lr", random.uniform, 1e-5, 1e-1)
toolbox.register("attr_float_momentum", random.uniform, 0.0, 0.9)
toolbox.register("attr_float_l2", random.uniform, 0.0001, 0.01)
toolbox.register("attr_float_decay", random.uniform, 1e-6, 1e-3)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_lr, toolbox.attr_float_momentum, toolbox.attr_float_l2, toolbox.attr_float_decay), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_model)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

logging.info("Initializing genetic algorithm for hyperparameter optimization.")
population = toolbox.population(n=10)
start_time = time.time()
result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
total_time = time.time() - start_time
logging.info(f"Evolutionary process completed in {total_time:.2f} seconds.")

best_ind = tools.selBest(population, 1)[0]
logging.info(f'Optimal Hyperparameters Found: Learning Rate = {best_ind[0]}, Momentum = {best_ind[1]}, L2 Reg = {best_ind[2]}, Decay = {best_ind[3]}')
