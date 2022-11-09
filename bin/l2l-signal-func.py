"""
This file is a typical example of a script used to run a L2L experiment. Read the comments in the file for more
explanations
"""
import shutil

import numpy as np

from l2l.optimizees.signal_generator_function.signal_generator import SignalGeneratorOptimizee
from l2l.optimizees.signal_generator_function.signal_generator import SignalGeneratorOptimizeeParameters
from l2l.optimizers.NN.optimizer import NNOptimizerParameters, NNOptimizer
from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
from l2l.optimizers.crossentropy.distribution import Gaussian, NoisyGaussian
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from l2l.optimizers.gradientdescent import RMSPropParameters, GradientDescentOptimizer, ClassicGDParameters, \
    StochasticGDParameters
from l2l.optimizers.gridsearch import GridSearchParameters, GridSearchOptimizer
from l2l.utils.experiment import Experiment
from l2l.optimizees.optimizee import Optimizee, OptimizeeParameters
from l2l.optimizers.optimizer import Optimizer, OptimizerParameters


def main(s1,s2):
    # define a directory to store the results
    experiment = Experiment(root_dir_path='../results')
    # prepare_experiment returns the trajectory and all jube parameters
    jube_params = {"nodes": "2",
                   "walltime": "10:00:00",
                   "ppn": "1",
                   "cpu_pp": "1"}
    traj, all_jube_params = experiment.prepare_experiment(name='L2L-SIGNAL-CE',
                                                          log_stdout=True,
                                                          debug=False)

    ## Innerloop simulator
    optimizee_parameters = SignalGeneratorOptimizeeParameters(frequency=5, amplitude=[1, 3], phase=[-1, 2], seed=2433,
                                                              range=1000)
    optimizee = SignalGeneratorOptimizee(traj, optimizee_parameters)

    # Cross Entropy ##
    # parameters = CrossEntropyParameters(pop_size=15, rho=0.15, smoothing=0.1, temp_decay=0, n_iteration=30,
    #                                     distribution=Gaussian(),
    #                                     stop_criterion=np.inf, seed=102)
    # optimizer = CrossEntropyOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                   optimizee_fitness_weights=(1.0,),
    #                                   parameters=parameters, optimizee_bounding_func=optimizee.bounding_func)

    # ---------------------    ### Gird Search ## ------------------------------#

    # Outerloop optimizer initialization
    # n_grid_divs_per_axis = 30
    # parameters = GridSearchParameters(param_grid={
    #     'amp': (-2, 2, n_grid_divs_per_axis), 'phase': (-2, 2, n_grid_divs_per_axis)
    # })
    # optimizer = GridSearchOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                 optimizee_fitness_weights=(1.0,),
    #                                 parameters=parameters)

    # ------------------------------------- GradientDescent ------------------------------------#

    # parameters = RMSPropParameters(learning_rate=0.025, exploration_step_size=0.02,
    #                                n_random_steps=9, momentum_decay=0.9,
    #                                n_iteration=50, stop_criterion=np.Inf, seed=s2)

    # parameters = AdaMaxParameters(learning_rate=0.02, exploration_step_size=0.02, n_random_steps=2, first_order_decay=0.9,
    #                             second_order_decay=0.999, n_iteration=15, stop_criterion=np.Inf,seed=123)

    # parameters = AdamParameters(learning_rate=0.02, exploration_step_size=0.02, n_random_steps=2, first_order_decay=0.9,
    #                             second_order_decay=0.999, n_iteration=15, stop_criterion=np.Inf,seed=123)

    # optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                      optimizee_fitness_weights=(1.0,),
    #                                      parameters=parameters)

    parameters = NNOptimizerParameters(learning_rate=0.001, pop_size=10, neurons=5, batch_size=512, epochs=100,
                                       input_path='../data_combined.csv', schema=[], header=0, target_category=0.9,
                                       n_iteration=10, stop_criterion=np.Inf, seed=6514)

    optimizer = NNOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                            optimizee_fitness_weights=(1.0,),
                            parameters=parameters)

    # ------------------------------------- Evol
    # tion ------------------------------------#
    # optimizer_seed = 1234
    # parameters = EvolutionStrategiesParameters(
    #     learning_rate=0.5,
    #     noise_std=1.0,
    #     mirrored_sampling_enabled=True,
    #     fitness_shaping_enabled=True,
    #     pop_size=10,
    #     n_iteration=20,
    #     stop_criterion=np.Inf,
    #     seed=optimizer_seed)
    #
    # optimizer = EvolutionStrategiesOptimizer(
    #     traj,
    #     optimizee_create_individual=optimizee.create_individual,
    #     optimizee_fitness_weights=1.0,
    #     parameters=parameters)

    # ------------------------------------- Genetic ------------------------------------#

    # parameters = GeneticAlgorithmParameters(seed=0, pop_size=10, cx_prob=0.5,
    #                                         mut_prob=0.3, n_iteration=20,
    #                                         ind_prob=0.02,
    #                                         tourn_size=15, mate_par=0.5,
    #                                         mut_par=1
    #                                         )
    #
    # optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                       optimizee_fitness_weights=(1),
    #                                       parameters=parameters)

    experiment.run_experiment(optimizee=optimizee,
                              optimizer=optimizer,
                              optimizer_parameters=parameters)
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    random = np.random.RandomState(5231)
    for i in range(1):
        shutil.rmtree('../results')
        seed1 = random.randint(1, 1000)
        seed2 = random.randint(1, 1000)
        main(seed1, seed2)
    # main()
