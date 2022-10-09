from datetime import datetime
from l2l.utils.experiment import Experiment
from l2l.optimizees.mc_gym.optimizee_mc import NeuroEvolutionOptimizeeMC, NeuroEvolutionOptimizeeMCParameters
from l2l.optimizers.evolution import GeneticAlgorithmParameters, GeneticAlgorithmOptimizer


def run_experiment():
    experiment = Experiment(root_dir_path='/p/scratch/icei-hbp-2022-0007/l2l_walid/')
    jube_params = {"exec": "srun -n 1 -c 1 --exact python"}
    traj, all_params = experiment.prepare_experiment(jube_parameter=jube_params, name="L2L-mc_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))
    # Inner-loop
    # Optimizee params
    optimizee_parameters = NeuroEvolutionOptimizeeMCParameters(
        path=experiment.root_dir_path, seed=1, save_n_generation=2, run_headless=True, load_parameter=False)
    optimizee = NeuroEvolutionOptimizeeMC(traj, optimizee_parameters)

    optimizer_seed = 12345678
    optimizer_parameters = GeneticAlgorithmParameters(seed=1580211, pop_size=120,
                                                      cx_prob=0.7,
                                                      mut_prob=0.7,
                                                      n_iteration=5,
                                                      ind_prob=0.45,
                                                      tourn_size=4,
                                                      mate_par=0.5,
                                                      mut_par=1
                                                      )

    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1,),
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=optimizee.bounding_func)
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()