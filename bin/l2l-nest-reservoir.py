import pickle 

from datetime import datetime
from l2l.utils.experiment import Experiment
from l2l.optimizees.snn.adaptive_optimizee import AdaptiveOptimizee, \
    AdaptiveOptimizeeParameters
from l2l.optimizers.kalmanfilter import EnsembleKalmanFilter, \
    EnsembleKalmanFilterParameters

def get_individuals(popsize, path, generation):
    res = []
    for i in range(popsize):
        with open(f'{path}/trajectory_{i}_{generation}.bin', 'rb') as t:
            indi = pickle.load(t)
            res.append(indi.individual)
    return res


def run_experiment():
    experiment = Experiment(root_dir_path='/p/scratch/icei-hbp-2021-0003/l2l_alper/')
    jube_params = {"exec": "srun -n 1 -c 8 --exclusive python"}
    traj, all_params = experiment.prepare_experiment(jube_parameter=jube_params, name="L2L-SNN-ENKF_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))

    # Optimizee params
    optimizee_parameters = AdaptiveOptimizeeParameters(
        path=experiment.root_dir_path,
        record_spiking_firingrate=True,
        save_plot=False)

    # Outer-loop optimizer initialization
    optimizer_seed = 1234
    pop_size = 98
    optimizer_parameters = EnsembleKalmanFilterParameters(gamma=0.5,
                                                          maxit=1,
                                                          n_iteration=400,
                                                          pop_size=pop_size,
                                                          n_batches=10,
                                                          n_repeat_batch=1,
                                                          online=False,
                                                          seed=optimizer_seed,
                                                          stop_criterion=1e-2,
                                                          path=experiment.root_dir_path,
                                                          scale_weights=True,
                                                          sample=True,
                                                          pick_method='random',
                                                          worst_n=0.11,
                                                          best_n=0.1,
                                                          data_loader_method='separated',
                                                          shuffle=True,
                                                          kwargs={'loc':0., 'scale':0.1}
                                                          )

    # Inner-loop simulator
    optimizee = AdaptiveOptimizee(traj, optimizee_parameters)
    # Outer-loop optimizer 
    optimizer = EnsembleKalmanFilter(traj,
                                     optimizee_prepare=optimizee.connect_network,
                                     optimizee_create_individual=optimizee.create_individual,
                                     optimizee_fitness_weights=(1.,),
                                     parameters=optimizer_parameters,
                                     optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer=optimizer,
                              optimizer_parameters=optimizer_parameters)

    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
