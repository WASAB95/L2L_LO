import os
import warnings
import logging.config

import numpy as np
from l2l.utils.environment import Environment

from l2l.optimizers.multigradientdescent.optimizer import MultiGradientDescentOptimizer
from l2l.optimizers.multigradientdescent.optimizer import MultiRMSPropParameters
from l2l.paths import Paths
from l2l.optimizees.pse_multi.optimizee import PSEOptimizee

import l2l.utils.JUBE_runner as jube
from l2l.utils.experiment import Experiment

#warnings.filterwarnings("ignore")

#logger = logging.getLogger('ltl-pse-multi-gd')


def main():
    name = 'LTL-PSE-MULTI-GD'

    jube_params = {"exec": "srun -n 1 --exclusive python"}

    experiment = Experiment(root_dir_path='../results')

    traj, _ = experiment.prepare_experiment(jube_parameter=jube_params, name=name, log_stdout=True)
    

    # NOTE: Innerloop simulator
    optimizee = PSEOptimizee(traj, seed=0)

    # NOTE: Outerloop optimizer initialization
    # TODO: Change the optimizer to the appropriate Optimizer class
    parameters = MultiRMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=4, momentum_decay=0.005,
                                   n_iteration=200, stop_criterion=np.Inf, seed=99, n_inner_params=1024)

    optimizer = MultiGradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee, optimizee_parameters=None, optimizer=optimizer, optimizer_parameters=parameters) 
    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()

