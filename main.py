import random
import time
import logging
import coloredlogs
from deap import base, creator, tools, algorithms
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import seaborn as sns

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
    momentum = min(max(0, momentum), 1)  # Ensure Momentum is in [0, 1]
    logger.debug(f"\033[1mConfiguring model with hyperparameters:\033[0m\n"
                 f" - Learning Rate: \033[34m{learning_rate:.3f}\033[0m\n"
                 f" - Momentum: \033[34m{momentum:.3f}\033[0m\n"
                 f" - L2 Regularization: \033[34m{l2_reg:.3f}\033[0m\n"
                 f" - Learning Rate Decay: \033[34m{lr_decay:.3f}\033[0m")

    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
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

# List to store stats across generations
accuracy_list = []
time_list = []
loss_list = []
fitness_list = []

def eval_model(hyperparameters):
    """
    Train the model and evaluate its performance on the validation dataset, with normalization of components to a [0, 1] scale.

    Parameters:
    - hyperparameters (tuple): The hyperparameters for the SGD optimizer.

    Returns:
    - tuple: A tuple containing one element, the normalized composite fitness score.
    """
    learning_rate, momentum, l2_reg, lr_decay = hyperparameters
    logger.info(f"\033[1mEvaluating model with hyperparameters:\033[0m\n"
                f" - Learning Rate: \033[34m{learning_rate:.3f}\033[0m\n"
                f" - Momentum: \033[34m{momentum:.3f}\033[0m\n"
                f" - L2 Regularization: \033[34m{l2_reg:.3f}\033[0m\n"
                f" - Learning Rate Decay: \033[34m{lr_decay:.3f}\033[0m")
    model = create_model(hyperparameters)
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, verbose=0, validation_split=0.1)
    elapsed_time = time.time() - start_time
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    logger.info(f"\033[1mTraining completed.\033[0m\n"
                f" - Time taken: \033[32m{elapsed_time:.2f} seconds\033[0m\n"
                f" - Validation accuracy: \033[32m{val_accuracy:.3f}\033[0m\n"
                f" - Validation loss: \033[32m{val_loss:.3f}\033[0m")

    # Normalizing each component
    max_time = 60  # Example: maximum observed time 60 seconds
    normalized_accuracy = val_accuracy
    normalized_time = (max_time - elapsed_time) / max_time
    normalized_loss = (1 - val_loss) / 1  # Assume maximum loss is 1

    # Composite fitness function
    fitness = 0.5 * normalized_accuracy + 0.3 * normalized_time + 0.2 * normalized_loss
    logger.info(f"\033[1mComposite Fitness Score:\033[0m \033[32m{fitness:.3f}\033[0m\n"
                f" - \033[1mWeighted Normalized Accuracy:\033[0m \033[32m{0.5 * normalized_accuracy:.3f}\033[0m\n"
                f" - \033[1mWeighted Normalized Time:\033[0m \033[32m{0.3 * normalized_time:.3f}\033[0m\n"
                f" - \033[1mWeighted Normalized Loss:\033[0m \033[32m{0.2 * normalized_loss:.3f}\033[0m")

    # Store the metrics
    accuracy_list.append(normalized_accuracy)
    time_list.append(elapsed_time)
    loss_list.append(val_loss)
    fitness_list.append(fitness)

    return (fitness,)

def plot_metrics():
    """
    Plot the metrics collected across generations.
    """
    generations = range(1, len(fitness_list) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.lineplot(x=generations, y=accuracy_list, marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 2)
    sns.lineplot(x=generations, y=time_list, marker='o')
    plt.title('Time Taken')
    plt.xlabel('Generation')
    plt.ylabel('Time (s)')

    plt.subplot(1, 3, 3)
    sns.lineplot(x=generations, y=loss_list, marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Generation')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.lineplot(x=generations, y=fitness_list, marker='o')
    plt.title('Composite Fitness Score')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.tight_layout()
    plt.show()

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
toolbox.register("mutGaussian", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

# Custom mutation function to ensure non-negative L2 Regularization and valid Momentum
def custom_mutate(individual):
    """
    Apply Gaussian mutation to the individual but ensure non-negative L2 Regularization and valid Momentum.
    """
    individual, = toolbox.mutGaussian(individual)
    mutation_index = random.randint(0, len(individual) - 1)
    logger.debug(f"\033[1;35mMutation applied at index {mutation_index}.\033[0m\n"
                 f" - Original: {individual}")
    individual[2] = max(0, individual[2])  # Ensure L2 Regularization is non-negative
    individual[1] = min(max(0, individual[1]), 1)  # Ensure Momentum is in [0, 1]
    logger.debug(f"\033[1;35mResult: {individual}\033[0m")
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
    logger.info("\033[1mEvaluated initial population.\033[0m")

    for gen in range(1, ngen + 1):
        gen_start_time = time.time()
        logger.info(f"\033[1mGeneration {gen} start.\033[0m")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                crossover_point = random.randint(1, len(child1) - 1)
                del child1.fitness.values
                del child2.fitness.values
                logger.debug(f"\033[1;34mCrossover applied at index {crossover_point}.\033[0m\n"
                             f" - Parent 1: {child1}\n"
                             f" - Parent 2: {child2}\n"
                             f" - Resulting Children: {child1}, {child2}")

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                logger.debug(f"\033[1;35mMutation applied to individual.\033[0m\n"
                             f" - Mutant: {mutant}")

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
        logger.info(f"\033[1mGeneration {gen}:\033[0m\n"
                    f" - Max: \033[32m{maximum:.3f}\033[0m\n"
                    f" - Min: \033[32m{minimum:.3f}\033[0m\n"
                    f" - Avg: \033[32m{mean:.3f}\033[0m\n"
                    f" - Time elapsed: \033[32m{time.time() - gen_start_time:.2f} seconds\033[0m")

    return population

logger.info("\033[1mInitializing genetic algorithm for hyperparameter optimization.\033[0m")
population = toolbox.population(n=10)
start_time = time.time()
result = custom_eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=3, verbose=True)
total_time = time.time() - start_time
logger.info(f"\033[1mEvolutionary process completed in\033[0m \033[32m{total_time:.2f} seconds\033[0m.")

best_ind = tools.selBest(population, 1)[0]
logger.info(f"\033[1mOptimal Hyperparameters Found:\033[0m\n"
            f" - Learning Rate: \033[34m{best_ind[0]:.3f}\033[0m\n"
            f" - Momentum: \033[34m{best_ind[1]:.3f}\033[0m\n"
            f" - L2 Regularization: \033[34m{best_ind[2]:.3f}\033[0m\n"
            f" - Learning Rate Decay: \033[34m{best_ind[3]:.3f}\033[0m")

plot_metrics()
