from datetime import datetime

from l2l.utils.experiment import Experiment
from l2l.optimizees.snn.optimizee_enkf import EnKFOptimizee, \
    EnKFOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters


def run_experiment():
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(jube_parameter={}, name="L2L-ENKF")

    # Inner-loop
    # Optimizee params
    optimizee_parameters = EnKFOptimizeeParameters(
        path=experiment.root_dir_path,
        stop_criterion=1e-2,
        record_spiking_firingrate=True,
        save_plot=False,
        n_test_batch=10,
        ensemble_size=10,
        n_batches=10,
        scale_weights=True,
        sample=True,
        pick_method='random',
        worst_n=0.11,
        best_n=0.1,
        data_loader_method='separated',
        shuffle=True,
        n_slice=4,
        kwargs={'loc': 0., 'scale': 0.1},
    )

    # Optimizee params
    # Inner-loop simulator
    optimizee = EnKFOptimizee(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    pop_size = 2
    optimizer_parameters = GeneticAlgorithmParameters(seed=0, pop_size=5,
                                                      cx_prob=0.7,
                                                      mut_prob=0.5,
                                                      n_iteration=3,
                                                      ind_prob=0.02,
                                                      tourn_size=2,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )

    optimizer = GeneticAlgorithmOptimizer(traj,
                                          optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters)
    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
