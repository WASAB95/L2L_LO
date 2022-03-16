from l2l.utils.experiment import Experiment
from l2l.optimizees.snn.optimizee_enkf import EnKFOptimizee, \
    EnKFOptimizeeParameters
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters


def run_experiment():
    experiment = Experiment(root_dir_path='../results')
    traj, _ = experiment.prepare_experiment(jube_parameter={}, name="L2L-ENKF")

    optimizee_parameters = EnKFOptimizeeParameters(
        record_spiking_firingrate=True,
        save_plot=False, n_test_batch=10, ensemble_size=10,
        n_batches=10, path=experiment.root_dir_path)
    # Optimizee params
    # Inner-loop simulator
    optimizee = EnKFOptimizee(traj, optimizee_parameters)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    pop_size = 2
    optimizer_parameters = GeneticAlgorithmParameters(seed=0, popsize=4,
                                                      CXPB=0.7,
                                                      MUTPB=0.5,
                                                      NGEN=3,
                                                      indpb=0.02,
                                                      tournsize=15,
                                                      matepar=0.5,
                                                      mutpar=1
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
