import shutil
from datetime import datetime

import numpy as np

from l2l.optimizers.NN.optimizer import NNOptimizerParameters, NNOptimizer
from l2l.utils.experiment import Experiment

from l2l.optimizees.mc_gym.optimizee_mc import NeuroEvolutionOptimizeeMC, NeuroEvolutionOptimizeeMCParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer

import os

def run_experiment(s1, s2):
    experiment = Experiment(
        root_dir_path='../results')
    jube_params = { "exec": "python3"}
    traj, _ = experiment.prepare_experiment(
         jube_parameter=jube_params, name="NeuroEvo_ES_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))
        
    # Optimizee params
    optimizee_parameters = NeuroEvolutionOptimizeeMCParameters(
        path=experiment.root_dir_path, seed=s1, save_n_generation=100, run_headless=True, load_parameter=False)
    optimizee = NeuroEvolutionOptimizeeMC(traj, optimizee_parameters)

    optimizer_seed = 12345678

    optimizer_parameters = NNOptimizerParameters(learning_rate=0.01, pop_size=10, neurons=5, batch_size=512, epochs=10,
                                       input_path='../data/all_data.csv', schema=[], header=0, target_category=0.45,
                                       n_iteration=20, stop_criterion=np.Inf, seed=s2)

    optimizer = NNOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                            optimizee_fitness_weights=(1.0,),
                            parameters=optimizer_parameters)

    # optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, pop_size=10,
    #                                                   cx_prob=0.7,
    #                                                   mut_prob=0.7,
    #                                                   n_iteration=10,
    #                                                   ind_prob=0.45,
    #                                                   tourn_size=4,
    #                                                   mate_par=0.5,
    #                                                   mut_par=1
    #                                                   )
    #
    # optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
    #                                       optimizee_fitness_weights=(1,),
    #                                       parameters=optimizer_parameters,
    #                                       optimizee_bounding_func=optimizee.bounding_func)
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)

def main(s1, s2):
    run_experiment(s1, s2)


if __name__ == '__main__':
    # shutil.rmtree('../results')
    random = np.random.RandomState(512)
    for i in range(5):
        shutil.rmtree('../results')
        seed1 = random.randint(1, 10000)
        seed2 = random.randint(1000, 5000)
        print(f"seed 1 {seed1}")
        print(f"seed 2 {seed2}")
        main(seed1, seed2)
    # main()
