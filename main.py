import random
import time
import logging
import coloredlogs
from deap import base, creator, tools, algorithms
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist

# Setup detailed logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
logger.info("Loading Fashion MNIST dataset.")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(hyperparameters):
    """
    Create and compile a neural network model based on specified SGD hyperparameters.

    Parameters:
    - hyperparameters (tuple): A tuple containing the hyperparameters to configure the model.

    Returns:
    - model (tf.keras.Model): The compiled neural network model.
    """
    learning_rate, momentum, l2_reg, lr_decay = hyperparameters
    l2_reg = max(0, l2_reg)  # Ensure L2 Regularization is non-negative
    logger.debug(f"Configuring model with hyperparameters: "
                 f"Learning Rate={learning_rate:.3f}, Momentum={momentum:.3f}, "
                 f"L2 Regularization={l2_reg:.3f}, Learning Rate Decay={lr_decay:.3f}")
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        Dense(10)
    ])
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100000,
        decay_rate=lr_decay,
        staircase=True
    )
    optimizer = SGD(learning_rate=learning_rate_schedule, momentum=momentum)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def eval_model(hyperparameters):
    """
    Train the model and evaluate its performance on the validation dataset.

    Parameters:
    - hyperparameters (tuple): The hyperparameters for the SGD optimizer.

    Returns:
    - tuple: A tuple containing one element, the negative of the validation accuracy.
    """
    logger.info(f"Evaluating model with hyperparameters: "
                f"Learning Rate={hyperparameters[0]:.3f}, Momentum={hyperparameters[1]:.3f}, "
                f"L2 Regularization={hyperparameters[2]:.3f}, Learning Rate Decay={hyperparameters[3]:.3f}")
    model = create_model(hyperparameters)
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, verbose=0, validation_split=0.1)
    elapsed_time = time.time() - start_time
    val_accuracy = history.history['val_accuracy'][-1]
    logger.info(f"Training completed in {elapsed_time:.2f} seconds. Validation accuracy: {val_accuracy:.3f}")
    return (-val_accuracy,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float_lr", random.uniform, 1e-5, 1e-1)
toolbox.register("attr_float_momentum", random.uniform, 0.0, 0.9)
toolbox.register("attr_float_l2", random.uniform, 0.0001, 0.01)
toolbox.register("attr_float_decay", random.uniform, 0.8, 1.0)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_lr, toolbox.attr_float_momentum, toolbox.attr_float_l2, toolbox.attr_float_decay), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_model)
toolbox.register("mate", tools.cxBlend, alpha=0.1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

# Custom mutation function to ensure non-negative L2 Regularization
def custom_mutate(individual):
    """
    Apply Gaussian mutation to the individual but ensure non-negative L2 Regularization.
    """
    toolbox.mutGaussian(individual, mu=0, sigma=1, indpb=0.1)
    individual[2] = max(0, individual[2])  # Ensure L2 Regularization is non-negative
    return (individual,)

toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def custom_eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    """
    Execute a simple evolutionary algorithm.

    Parameters:
    - population: A list of individuals representing the population.
    - toolbox: A DEAP toolbox containing the genetic operations.
    - cxpb: The crossover probability.
    - mutpb: The mutation probability.
    - ngen: The number of generations.
    - stats: A DEAP statistics object.
    - halloffame: A DEAP Hall of Fame object.
    - verbose: A flag indicating verbosity.

    Returns:
    - population: The final population after all generations.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    logger.info("Evaluated initial population.")

    for gen in range(1, ngen + 1):
        gen_start_time = time.time()
        logger.info(f"Generation {gen} start.")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                logger.debug("Crossover applied.")

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                logger.debug("Mutation applied.")

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        logger.info(f"Evaluated {len(invalid_ind)} individuals.")

        # Replace the old population with the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        maximum = max(fits)
        minimum = min(fits)
        mean = sum(fits) / len(fits)
        logger.info(f"Generation {gen}: Max {maximum:.3f}, Min {minimum:.3f}, Avg {mean:.3f}. "
                    f"Time elapsed: {time.time() - gen_start_time:.2f} seconds.")

    return population

logger.info("Initializing genetic algorithm for hyperparameter optimization.")
population = toolbox.population(n=10)
start_time = time.time()
result = custom_eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
total_time = time.time() - start_time
logger.info(f"Evolutionary process completed in {total_time:.2f} seconds.")

best_ind = tools.selBest(population, 1)[0]
logger.info(f'Optimal Hyperparameters Found: Learning Rate = {best_ind[0]:.3f}, Momentum = {best_ind[1]::.3f}, '
            f'L2 Regularization = {best_ind[2]:.3f}, Learning Rate Decay = {best_ind[3]:.3f}')
