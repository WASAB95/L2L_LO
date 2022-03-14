from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter
from l2l.optimizers.kalmanfilter.data import fetch
from l2l.optimizees.snn.reservoir_nest3 import Reservoir
from scipy.special import softmax

import json
import numpy as np
import os
import pathlib
import pandas as pd
import subprocess
import time

# TODO:
'''
- add sampling
- make test static
- weights are saved as `{individual_index}{simulation_index}_weights_{type}.csv`
- the key in the dictionary is `{individual_index}{simulation_index}_weights_{type}`
- how to return the fitness (best, mean)? 
'''

EnKFOptimizeeParameters = namedtuple(
    'EnKFOptimizeeParameters', ['path',    # path to the result files
                                'record_spiking_firingrate',
                                'save_plot',
                                'replace_weights',
                                'n_batch',    # n images to be shown
                                'n_test_batch',    # n test images to be shown
                                ])


class EnKFOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.parameters = parameters
        fp = pathlib.Path(__file__).parent.absolute()
        print(os.path.join(str(fp), 'config.json'))
        with open(
                os.path.join(str(fp), 'config.json')) as jsonfile:
            self.config = json.load(jsonfile)
        # Lists for labels
        self.target_labels = []
        self.random_ids = []
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx

        # MNIST DATA HANDLING
        self.target_label = ['0', '1']
        self.other_label = ['2', '3', '4', '5', '6', '7', '8', '9']
        self.train_set = None
        self.train_labels = None
        self.other_set = None
        self.other_labels = None
        self.test_set = None
        self.test_labels = None
        self.test_set_other = None
        self.test_labels_other = None
        self.optimizee_labels = None
        self.reservoir = Reservoir()
        self.rng = np.random.default_rng(self.gen_idx)

        # dictionary for weights
        self.dict_weights = {}
        # batch file
        self.batchfile = 'batchfile.sh'
        self.types = ['eeo', 'eio', 'ieo', 'iio']

        self.gamma = 0.
        self.ensemble_size = 0
        self.repetitions = 0

    def get_mnist_data(self, mnist_path='./mnist784_dat/'):
        self.train_set, self.train_labels, self.test_set, self.test_labels = \
            fetch(path=mnist_path, labels=self.target_label)
        self.other_set, self.other_labels, self.test_set_other, self.test_labels_other = \
            fetch(path=mnist_path, labels=self.other_label)

    @staticmethod
    def randomize_labels(labels, size):
        """
        Randomizes given labels `labels` with size `size`.

        :param labels: list of strings with labels
        :param size: int, size of how many labels should be returned
        :return list of randomized labels
        :return list of random numbers used to randomize the `labels` list
        """
        rnd = np.random.randint(low=0, high=len(labels), size=size)
        return [int(labels[i]) for i in rnd], rnd

    def create_batchfile(self, csv_path):
        with open(os.path.join(csv_path, self.batchfile), 'w') as f:
            batch = "#!/bin/bash -x \n" \
                    "#SBATCH --nodes=2 \n" \
                    "#SBATCH --ntasks-per-node=128\n" \
                    "#SBATCH --cpus-per-task=1 \n" \
                    "#SBATCH --time=1:00:00 \n" \
                    "#SBATCH --partition=batch #SBATCH --account=slns \n " \
                    "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} \n " \
                    f"srun python reservoir.py $1 $2 $3 $4"
            f.write(batch)

    def create_ensembles(self):
        # connect network
        self.execute_subprocess(self.parameters.path, simulation='--create')
        while True:
            if all([os.path.isfile(
                    os.path.join(self.parameters.path, f'{t}_connections.csv'))
                    for t in self.types]):
                break
            else:
                time.sleep(3)

        mu = self.config['mu']
        sigma = self.config['sigma']
        size_eeo, size_eio, size_ieo, size_iio = self.get_connections_sizes(self.parameters.path)
        for i in range(self.ensemble_size):
            self.dict_weights[
                f'{self.ind_idx}{i}_weights_eeo'] = np.random.normal(mu, sigma,
                                                                     size_eeo)
            self.dict_weights[
                f'{self.ind_idx}{i}_weights_eio'] = np.random.normal(mu, sigma,
                                                                     size_eio)
            self.dict_weights[
                f'{self.ind_idx}{i}_weights_ieo'] = np.random.normal(-mu,
                                                                     sigma,
                                                                     size_ieo)
            self.dict_weights[
                f'{self.ind_idx}{i}_weights_iio'] = np.random.normal(-mu,
                                                                     sigma,
                                                                     size_iio)
            # finally save weights
            self.save_weights(self.parameters.path, i)

    def execute_subprocess(self, csv_path, index='00', simulation='--create'):
        if simulation == '--create' or simulation == '-c':
            sub = subprocess.run(
                ['sbatch', f'{simulation}',
                 '-p', f'{csv_path}',
                 '-g', f'{self.gen_idx}'],
                check=True)
        else:
            sub = subprocess.run([
                'sbatch', f'{simulation}',
                '--index', f'{index}',
                '--generation', f'{self.gen_idx}',
                '--path', f'{csv_path}',
                '--record_spiking_firingrate', f'{self.parameters.record_spiking_firingrate}',
            ],
                check=True)
        sub.check_returncode()

    def create_individual(self):
        return {'gamma': np.abs(self.rng.normal(0., 0.1, size=1)),
                'ensemble_size': self.rng.integers(10, 32, size=1),
                'repetitions': self.rng.integers(1, 5, size=1),
                # TODO sampling values
                }

    def bounding_func(self, individual):
        self.gamma = np.clip(individual['gamma'], 0.01, 1.)
        self.ensemble_size = np.clip(individual['ensemble_size'], 10, 32).astype(int)
        self.repetitions = np.clip(individual['repetitions'], 1, 5).astype(int)
        individual = {'gamma': self.gamma,
                      'ensemble_size': self.ensemble_size,
                      'repetitions': self.repetitions,
                      }
        return individual

    def simulate(self, traj):
        # get indices
        self.gen_idx = traj.individual.generation
        self.ind_idx = traj.individual.ind_idx
        self.rng = np.random.default_rng(self.gen_idx)
        self.gamma = traj.individual.gamma
        self.repetitions = traj.individual.repetitions
        self.ensemble_size = traj.individual.ensemble_size
        if self.gen_idx == 0:
            # create the batch file if non existent
            if not os.path.isfile(
                    os.path.join(self.parameters.path, self.batchfile)):
                self.create_batchfile(csv_path=self.parameters.path)
            self.create_ensembles()
        # load the latest weights
        for i in range(self.ensemble_size):
            index = f'{self.ind_idx}{i}'
            self.load_weights(csv_path=self.parameters.path,
                              simulation_idx=index)
        # get new data
        self.get_mnist_data()
        if self.train_labels:
            self.optimizee_labels, self.random_ids = self.randomize_labels(
                self.train_labels, size=self.parameters.n_test_batch)
            self.train_set = [self.train_set[r] for r in self.random_ids]
            # save train set and train labels
            # all individuals/simulations are getting the same batch of data
            self.save_data_set(file_path=self.parameters.path,
                               trainset=self.train_set,
                               targets=self.optimizee_labels,
                               generation=self.gen_idx)

        # Prepare for simulation
        n_output_clusters = self.config['n_output_clusters']
        enkf = EnsembleKalmanFilter(maxit=1,
                                    online=True,
                                    n_batches=self.parameters.n_batch)
        # Show i times the batch of images
        for i in range(self.repetitions):
            indices = []
            for j in range(self.ensemble_size):
                # save weights before simulation
                self.save_weights(csv_path=self.parameters.path,
                                  simulation_idx=j)
                index = f'{self.ind_idx}{j}'
                self.execute_subprocess(csv_path=self.parameters.path,
                                        index=index, simulation='--simulate')
                indices.append(index)
            while True:
                if all([os.path.isfile(os.path.join(self.parameters.path,
                                                    f'{idx}_model_out.csv'))
                        for idx in indices]):
                    break
                else:
                    time.sleep(3)

            model_outs = [pd.read_csv(os.path.join(
                self.parameters.path, f'{idx}_model_out.csv'))['model_out'].values
                          for idx in indices]
            # obtain the individual model outputs and apply softmax on them
            model_outs, argmax = self.apply_softmax(model_outs=model_outs,
                                                    n_output_clusters=n_output_clusters)
            # EnKF fit
            # concatenate weights
            ens = [np.concatenate(
                (self.dict_weights[f'{self.ind_idx}{i}_weights_eeo'],
                 self.dict_weights[f'{self.ind_idx}{i}_weights_eio'],
                 self.dict_weights[f'{self.ind_idx}{i}_weights_ieo'],
                 self.dict_weights[f'{self.ind_idx}{i}_weights_iio']))
                for i in range(self.ensemble_size)]
            enkf.fit(ensemble=np.array(ens),
                     model_output=np.array(model_outs),
                     ensemble_size=self.ensemble_size,
                     observations=np.array(self.optimizee_labels),
                     gamma=self.gamma)

            self.restructure_weight_dict(enkf.ensemble.cpu().numpy(),
                                         csv_path=self.parameters.path,
                                         simulation_index=i)
        # save new weights after optimization is done
        for i in range(self.ensemble_size):
            self.save_weights(csv_path=self.parameters.path, simulation_idx=i)

        # Final testing to obtain the fitness
        if self.test_labels:
            self.optimizee_labels, self.random_ids = self.randomize_labels(
                self.test_labels, size=self.parameters.n_test_batch)
            self.test_set = [self.test_set[r] for r in self.random_ids]
            # save test set and test labels
            self.save_data_set(file_path=self.parameters.path,
                               trainset=self.train_set,
                               targets=self.optimizee_labels,
                               generation=self.gen_idx)

        indices = []
        for j in range(self.ensemble_size):
            index = f'{self.ind_idx}{j}'
            self.execute_subprocess(csv_path=self.parameters.path,
                                    index=index, simulation='--simulate')
            indices.append(index)
        while True:
            if all([os.path.isfile(os.path.join(self.parameters.path,
                                                f'{idx}_model_out.csv'))
                    for idx in indices]):
                break
            else:
                time.sleep(3)
        model_outs = [pd.read_csv(os.path.join(
            self.parameters.path, f'{idx}_model_out.csv'))['model_out'].values
                      for idx in indices]
        model_outs, argmax = self.apply_softmax(model_outs=model_outs,
                                                n_output_clusters=n_output_clusters)
        fitnesses = []
        # one hot encoding
        for i, target in enumerate(self.optimizee_labels):
            label = np.eye(n_output_clusters)[target]
            pred = np.eye(n_output_clusters)[i]
            # MSE of 1 is worst 0 is best, that's why 1 - mean
            fitness = 1 - self._calculate_fitness(label, pred, "MSE")
            fitnesses.append(fitness)
            print('Fitness {} for target {}, softmax {}, argmax {}'.format(
                fitness, target, model_outs[i], argmax[i]))
        self.dict_weights.clear()
        return (np.mean(fitnesses),)

    def save_weights(self, csv_path, simulation_idx):
        # Read the connections
        for typ in self.types:
            conns = pd.read_csv(os.path.join(csv_path,
                                             '{}_connections.csv'.format(typ)))
            # add new columns of weights
            key = f'{self.ind_idx}{simulation_idx}_weights_{typ}'
            conns['weights'] = self.dict_weights.get(key)
            # remove if file with weights exists already
            if os.path.exists(os.path.join(csv_path, key, '.csv')):
                os.remove(os.path.join(csv_path, key))
            # write file with connections and new weights
            conns.to_csv(os.path.join(csv_path, f'{key}.csv'))

    def load_weights(self, csv_path, simulation_idx):
        for typ in self.types:
            conns = pd.read_csv(os.path.join(csv_path,
                                             f'{self.ind_idx}{simulation_idx}_weights_{typ}.csv'))
            key = f'{self.ind_idx}{simulation_idx}_weights_{typ}'
            self.dict_weights[key] = conns['weights'].values

    def restructure_weight_dict(self, w, csv_path, simulation_index):
        length = 0
        for typ in self.types:
            conns = pd.read_csv(os.path.join(csv_path,
                                             '{}_connections.csv'.format(typ)))
            key = f'{self.ind_idx}{simulation_index}_weights_{typ}'
            self.dict_weights[key] = w[length:len(conns['sources']) + length]
            length = len(conns['sources'])

    @staticmethod
    def save_data_set(file_path, trainset, targets, generation):
        """
        Saves the dataset and label as a numpy file per generation.
        If the file already exists the data is not created.
        """
        filename = f"{generation}_dataset.npy"
        if not os.path.exists(filename):
            np.save(os.path.join(file_path, filename),
                    {'train_set': trainset, 'targets': targets},
                    allow_pickle=True)

    def get_connections_sizes(self, csv_path):
        sizes = []
        for typ in self.types:
            con = pd.read_csv(os.path.join(
                csv_path, '{}_connections.csv'.format(typ)))
            sizes.append(len(con['source']))
        return tuple(sizes)

    @staticmethod
    def apply_softmax(model_outs, n_output_clusters):
        mos = []
        argmax = []
        # obtain the individual model outputs and apply softmax on them
        for model_out in model_outs:
            softm = softmax([np.mean(model_out[j]) for j in
                             range(n_output_clusters)])
            argmax.append(np.argmax(softm))
            mos.append(softm)
        return mos, argmax

    @staticmethod
    def _calculate_fitness(label, prediction, costf='MSE'):
        if costf == 'MSE':
            return ((label - prediction) ** 2).mean()


def randomize_labels(labels, size):
    rnd = np.random.randint(low=0, high=len(labels), size=size)
    return [int(labels[i]) for i in rnd], rnd


if __name__ == "__main__":
    from l2l import DummyTrajectory
    from l2l.paths import Paths
    from l2l import sdict

    root_dir_path = "../../../results"
    if not os.path.exists(root_dir_path):
        os.mkdir(root_dir_path)
    paths = Paths('', {}, root_dir_path=root_dir_path)

    fake_traj = DummyTrajectory()
    fake_traj.individual.generation = 0
    fake_traj.individual.ind_idx = 0
    # Optimizee params
    optimizee_parameters = EnKFOptimizeeParameters(
        path=paths.root_dir_path,
        record_spiking_firingrate=True,
        save_plot=False, ensemble_size=32, n_simulations=5, n_test_batch=10,
        n_batch=10, replace_weights=False, repetitions=2, gamma=0.01, seed=0)
    # Inner-loop simulator
    optimizee = EnKFOptimizee(fake_traj, optimizee_parameters)
    print(optimizee.config)
    # size_eeo, size_eio, size_ieo, size_iio = optimizee.connect_network()
    #print(size_eeo, size_eio, size_ieo, size_iio,
    #      f'total size {size_eeo+size_eio+size_ieo+size_iio}')
    # params = optimizee.create_individual(
    #     size_eeo, size_eio, size_ieo, size_iio)
    path = '/home/yegenoglu/Documents/toolbox/results_l2l/SNN-EnKF/2021-05-06/'
    # load all weights,  select iteration and individual
    weights = np.load(os.path.join(path, 'weights.npz')
                      ).get('weights')[350][0]
    # optimizee.replace_weights_(weights, path=root_dir_path)
    grouped_params_dict = {}  # {key: val for key, val in params.items()}
    grouped_params_dict['generation'] = 0
    grouped_params_dict['ind_idx'] = 0

    # MNIST DATA HANDLING
    target_label = ['2']
    other_label = ['0', '1', '3', '4', '5', '6', '7', '8', '9']

    # get the targets
    train_set, train_labels, test_set, test_labels = \
        fetch(path='./mnist784_dat/',
                   labels=target_label)
    other_set, other_labels, test_set_other, test_labels_other = fetch(
        path='./mnist784_dat/', labels=other_label)

    if train_labels:
        n_batch = 10
        optimizee_labels, random_ids = randomize_labels(
            test_labels, size=n_batch)
    else:
        raise AttributeError('Train Labels are not set, please check.')

    grouped_params_dict["targets"] = optimizee_labels
    print(optimizee_labels)
    # t = [test_set[r] for r in random_ids]
    # print(t)
    grouped_params_dict["train_set"] = test_set
    fake_traj.individual = sdict(grouped_params_dict)
    testing_error = optimizee.simulate(fake_traj)
    print(testing_error)
