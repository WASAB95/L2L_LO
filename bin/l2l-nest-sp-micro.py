from l2l.utils.experiment import Experiment

from l2l.optimizers.crossentropy.optimizer import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian
from l2l.optimizers.evolutionstrategies.optimizer import EvolutionStrategiesOptimizer, EvolutionStrategiesParameters
from l2l.optimizers.gradientdescent.optimizer import GradientDescentOptimizer, RMSPropParameters

from l2l.optimizees.sp_micro.optimizee import SPMicrocircuitOptimizee

from datetime import datetime

def run_experiment():
    experiment = Experiment(root_dir_path='./results')
    jube_params = {"exec": "srun -n 4 -c 1 --exclusive python"}
    traj, all_params = experiment.prepare_experiment(jube_parameter=jube_params, name="L2L-SP-MICRO_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))

    # Optimizer params
    optimizer_parameters = RMSPropParameters(learning_rate=0.00005, exploration_step_size=0.001,
                                   n_random_steps=9, momentum_decay=0.005,
                                   n_iteration=100, stop_criterion=np.Inf, seed=99)

    # Inner-loop simulator
    optimizee = SPMicrocircuitOptimizee(traj, seed=0)
    
    # Outer-loop optimizer 
    optimizer = GradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(1.0,),
                                         parameters=optimizer_parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
