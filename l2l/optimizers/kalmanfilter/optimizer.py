import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.preprocessing as pp

from collections import namedtuple
from l2l.optimizers.kalmanfilter.enkf import EnsembleKalmanFilter as EnKF
from l2l import dict_to_list
from l2l.optimizers.optimizer import Optimizer
from l2l.optimizers.crossentropy.distribution import Gaussian

from l2l.optimizers.kalmanfilter import data

logger = logging.getLogger("optimizers.kalmanfilter")

EnsembleKalmanFilterParameters = namedtuple(
    'EnsembleKalmanFilter', ['gamma', 'maxit', 'n_iteration',
                             'pop_size', 'n_batches', 'online', 'seed', 'path',
                             'n_repeat_batch', 'stop_criterion',
                             'scale_weights', 'sample',
                             'best_n', 'worst_n', 'pick_method',
                             'data_loader_method', 'shuffle', 'n_slice',
                             'kwargs'],
    defaults=(True, False, 0.25, 0.25, 'random', 'separated', False, 4,
              {'pick_probability': 0.7, 'loc': 0, 'scale': 0.1})
)

EnsembleKalmanFilterParameters.__doc__ = """
:param gamma: float, A small value, multiplied with the eye matrix
:param maxit: int, Epochs to run inside the Kalman Filter
:param n_iteration: int, Number of iterations to perform
:param pop_size: int, Minimal number of individuals per simulation.
    Corresponds to number of ensembles
:param n_batches: int, Number of mini-batches to use for training
param n_repeat_batch: int, How often the same image batch should be shown
:param online: bool, Indicates if only one data point will used
    Default: False
:param scale_weights: bool, scales weights between [0, 1]
:param sampling: bool, If sampling of best individuals should be done
:param best_n: float, Percentage for best `n` individual. In combination with
    `sampling`. Default: 0.25
:param worst_n: float, Percentage for worst `n` individual. In combination with
    `sampling`. Default: 0.25
:param pick_method: str, How the best individuals should be taken. `random` or
    `best_first` must be set. If `pick_probability` is taken then a key
    word argument `pick_probability` with a float value is needed. `gaussian` 
    creates a multivariate normal distribution using the best individuals 
    which will replace the worst individuals. (see also :param kwargs) 
    In combination with `sampling`.
    Default: 'random'.
:param data_loader_method: str, Method on how to load the data set. 
    - `random`: Just randomly pick `n_batch` images from the whole data set.
    - 'separated': Separates the data set according to the test labels,
        e.g. [1,1,2,0,0,2] -> [[0,0],[1,1],[2,2]]. 
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param shuffle: bool, if `True` shuffles the data and labels. In combination 
    with `data_loader_method`.  
        Default is False. 
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param n_slice: int, How many slices are going to be taken.
        E.g. if target labels are `[0,1,2]` and `n_slice` 4 then the
        function will return `3 * 4 = 12` labels and corresponding data back. 
        The slice starts at the generation number. Used in combination with
        `shuffle=True`. Default is 4.
        See also l2l.optimizers.kalmanfilter.get_separated_data
:param kwargs: dict, key word arguments if `sampling` is True.
    - `pick_probability` - float, probability to pick the first best individual
      Default: 0.7
    - loc - float, mean of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability`
      Default: 0.
    - scale - float, std scale of the gaussian normal, can be specified if 
      `pick_method` is `random` or `pick_probability` 
      Default: 0.1
:param seed: The random seed used to sample and fit the distribution. 
    Uses a random generator seeded with this seed.
:param path: String, Root path for the file saving and loading the connections
:param stop_criterion: float, When the current fitness is smaller or equal the 
    `stop_criterion` the optimization in the outer loop ends
"""


class EnsembleKalmanFilter(Optimizer):
    """
    Class for an Ensemble Kalman Filter optimizer
    """

    def __init__(self, traj,
                 optimizee_prepare,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):
        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters,
                         optimizee_bounding_func=optimizee_bounding_func)

        self.optimizee_bounding_func = optimizee_bounding_func
        self.optimizee_create_individual = optimizee_create_individual
        self.optimizee_fitness_weights = optimizee_fitness_weights
        self.optimizee_prepare = optimizee_prepare

        traj.f_add_parameter('gamma', parameters.gamma, comment='Noise level')
        traj.f_add_parameter('maxit', parameters.maxit,
                             comment='Maximum iterations')
        traj.f_add_parameter('n_iteration', parameters.n_iteration,
                             comment='Number of iterations to run')
        traj.f_add_parameter('n_batches', parameters.n_batches)
        traj.f_add_parameter('n_repeat_batch', parameters.n_repeat_batch)
        traj.f_add_parameter('online', parameters.online)
        traj.f_add_parameter('seed', np.uint32(parameters.seed),
                             comment='Seed used for random number generation '
                                     'in optimizer')
        traj.f_add_parameter('pop_size', parameters.pop_size)
        traj.f_add_parameter('path', parameters.path,
                             comment='Root folder for the simulation')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion,
                             comment='stopping threshold')
        traj.f_add_parameter('sample', parameters.sample,
                             comment='sampling on/off')
        traj.f_add_parameter('data_loader_method', parameters.data_loader_method,
                             comment='How to load data')
        traj.f_add_parameter('shuffle', parameters.shuffle,
                             comment='Shuffles the data')
        traj.f_add_parameter('n_slice', parameters.n_slice,
                             comment='Take n slices of data')
        traj.f_add_parameter('scale_weights', parameters.scale_weights,
                             comment='scaling of weights')
        if parameters.sample:
            traj.f_add_parameter('best_n', parameters.best_n,
                                 comment='best n individuals')
            traj.f_add_parameter('worst_n', parameters.worst_n,
                                 comment='worst n individuals')
            traj.f_add_parameter('pick_method', parameters.pick_method,
                                 comment='how to pick random individual')
            traj.f_add_parameter('kwargs', parameters.kwargs,
                                 comment='dict with key word arguments')

        #: The population (i.e. list of individuals) to be evaluated at the
        # next iteration
        size_eeo, size_eio, size_ieo, size_iio = self.optimizee_prepare()
        _, self.optimizee_individual_dict_spec = dict_to_list(
            self.optimizee_create_individual(
                size_eeo, size_eio, size_ieo, size_iio),
            get_dict_spec=True)

        traj.results.f_add_result_group('generation_params')

        # Set the random state seed for distribution
        self.random_state = np.random.RandomState(traj.parameters.seed)

        current_eval_pop = [self.optimizee_create_individual(size_eeo,
                                                             size_eio,
                                                             size_ieo,
                                                             size_iio)
                            for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            pass
            # current_eval_pop = [self.optimizee_bounding_func(ind) for ind in
            #                     current_eval_pop]

        self.eval_pop = current_eval_pop
        self.best_individual = []
        self.current_fitness = np.inf
        self.fitness_all = []
        # weights to save pro certain generation for further analysis
        self.weights_to_save = []
        self.cov_to_save = []

        # self.targets = parameters.observations

        self.g = traj.individual.generation # 0
        # MNIST DATA HANDLING
        self.target_label = ['0', '1', '2']
        self.other_label = ['3', '4', '5', '6', '7', '8', '9']
        self.train_set = None
        self.train_labels = None
        self.other_set = None
        self.other_labels = None
        self.test_set = None
        self.test_labels = None
        self.test_set_other = None
        self.test_labels_other = None
        # get the targets
        self.get_mnist_data()
        self.data_loader_method = traj.parameters.data_loader_method
        if self.train_labels:
            if self.data_loader_method == 'random':
                self.optimizee_labels, self.random_ids = self.randomize_labels(
                    self.train_labels, size=traj.n_batches)
            elif self.data_loader_method == 'separated':
                self.optimizee_data, self.optimizee_labels = self.get_separated_data(
                    self.train_labels, self.train_set, self.target_label,
                    shuffle=traj.parameters.shuffle,
                    n_slice=traj.parameters.n_slice)
                if not traj.parameters.shuffle:
                    rand_ind = self.random_state.randint(0, len(self.target_label), 1)[0]
                    self.optimizee_labels, self.random_ids = self.randomize_labels(
                        self.optimizee_labels[rand_ind], size=traj.n_batches)
        else:
            raise AttributeError('Train Labels are not set, please check.')
        logger.info('First dataset is set. Targets are {}'.format(
            self.optimizee_labels))

        for e in self.eval_pop:
            if self.data_loader_method == 'random':
                e["targets"] = self.optimizee_labels
                e["train_set"] = [self.train_set[r] for r in self.random_ids]
            elif self.data_loader_method == 'separated':
                if traj.parameters.shuffle:
                    e["targets"] = self.optimizee_labels
                    e["train_set"] = self.optimizee_data
                else:
                    e["targets"] = self.optimizee_labels
                    e["train_set"] = [self.optimizee_data[rand_ind][r] for r in
                                      self.random_ids]

        self._expand_trajectory(traj)

    def get_mnist_data(self):
        self.train_set, self.train_labels, self.test_set, self.test_labels = \
            data.fetch(path='./mnist784_dat/',
                       labels=self.target_label)
        self.other_set, self.other_labels, self.test_set_other, self.test_labels_other = data.fetch(
            path='./mnist784_dat/', labels=self.other_label)

    def randomize_labels(self, labels, size):
        """
        Randomizes given labels `labels` with size `size`.

        :param labels: list of strings with labels
        :param size: int, size of how many labels should be returned
        :return list of randomized labels
        :return list of random numbers used to randomize the `labels` list
        """
        rnd = self.random_state.randint(low=0, high=len(labels), size=size)
        return [int(labels[i]) for i in rnd], rnd

    def get_separated_data(self, train_labels, train_set, target_labels,
                           shuffle=False, gen_id=0, n_slice=4):
        """
        Separates the data set according to the test labels,
        e.g. [1,1,2,0,0,2] -> [[0,0],[1,1],[2,2]].
        It returns the separated data set and labels.

        :param train_labels: array_like, labels of the data set
        :param train_set: array_like, the data set itself
        :param target_labels: array_like, the labels/digits which are going to
            be separated
        :param gen_id: int, the generation index. Used in combination with `
            shuffle=True`. Default is 0.
        :param n_slice: int, how many slices are going to be taken.
            E.g. if target labels are `[0,1,2]` and `n_slice` 4 then the
            function will return `3 * 4 = 12` labels and corresponding data
            back. The slice starts at `gen_id * n_slice`. Used in combination
            with `shuffle=True`. Default is 4.
        :param shuffle: bool, if `True` shuffles the data and labels.
            Default is False.
        """
        data_set = []
        data_labels = []
        for tl in target_labels:
            indices = np.array(train_labels) == tl
            data_set.append(np.array(train_set)[indices])
            data_labels.append(np.array(train_labels)[indices])
        if shuffle:
            sliced_data = []
            sliced_labels = []
            index0 = gen_id * n_slice
            index1 = gen_id * n_slice + n_slice
            for i in range(len(target_labels)):
                len_data = len(data_set[i])
                sliced_data.append(data_set[i][index0 % len_data: index1 % len_data])
                len_labels = len(data_labels[i])
                sliced_labels.append(data_labels[i][index0 % len_labels: index1 % len_labels])
            # shuffle now
            sliced_labels = np.array(sliced_labels).ravel()
            shuffled_index = np.arange(sliced_labels.size)
            self.random_state.shuffle(shuffled_index)
            # reshape so that the shuffling is possible according the index
            sliced_data = np.reshape(sliced_data, (shuffled_index.size, -1))
            return sliced_data[shuffled_index], sliced_labels[shuffled_index].astype(int)
        else:
            return data_set, data_labels

    @staticmethod
    def _get_every_n_data(data_set, targets, gen_id, random_state, n_data=10, shuffle=True):
        """
        Return every `n_data` data element, according to the generation
        index `gen_id`. Additionally, it makes sure not to index beyond the
        inidividual data elements length.

        Notes:
        Requires separated dataset to work.
        Should be used in combination with `get_separated_data` function.

        :param data_set: array_like, the data set as a list or nd.array
        :param gen_id: int, the generation index
        :param targets all labels which correspond to the data set
        :param n_data: int, the number of data elements to be returned
            Default is 10.
        :param shuffle: bool, If ``True`` then the function returns a
            shuffled version of the data set and targets.
            Default is True.
        """
        # get first the indices
        index0 = gen_id * n_data
        index1 = gen_id * n_data + n_data
        nth_data = []
        nth_label = []
        for ds, targ in zip(data_set, targets):
            # modulo operation to not go beyond the total length of the
            # data element
            nth_data.extend(ds[index0 % len(ds): index1 % len(ds)])
            nth_label.extend(targ[index0 % len(targ): index1 % len(targ)])
        if shuffle:
            shuffled_index = np.arange(len(nth_label))
            random_state.shuffle(shuffled_index)
            return np.array(nth_data)[shuffled_index], np.array(nth_label)[shuffled_index].astype(int)
        else:
            return np.array(nth_data), np.array(nth_label).astype(int)

    def post_process(self, traj, fitnesses_results):
        self.eval_pop.clear()

        individuals = traj.individuals[self.g]
        gamma = np.eye(len(self.target_label)) * traj.gamma

        ensemble_size = traj.pop_size
        # before scaling the weights, check for the shapes and adjust
        # with `_sample_from_individual`
        # weights = [traj.current_results[i][1]['connection_weights'] for i in
        #           range(ensemble_size)]
        weights = [np.concatenate(
            (individuals[i].weights_eeo, individuals[i].weights_eio,
             individuals[i].weights_ieo,  individuals[i].weights_iio))
            for i in range(ensemble_size)]
        fitness = [traj.current_results[i][1]['fitness'] for i in
                   range(ensemble_size)]
        self.current_fitness = np.max(fitness)

        ens = np.array(weights)
        model_outs = np.array([traj.current_results[i][1]['model_out'] for i in
                               range(ensemble_size)])
        if traj.scale_weights:
            # (ens - ens.min()) / (ens.max() - ens.min())
            ens = ens / np.abs(ens).max()
            # ens, scaler = self._scale_weights(weights, normalize=True,
            #                                   method=pp.MinMaxScaler)

        # sampling step
        # if traj.sample:
        #     ens, model_outs = self.sample_from_individuals(individuals=ens,
        #                                                    model_output=model_outs,
        #                                                    fitness=fitness,
        #                                                    sampling_method=traj.sampling_method,
        #                                                    pick_method=traj.pick_method,
        #                                                    best_n=traj.best_n,
        #                                                    worst_n=traj.worst_n / np.exp(
        #                                                        self.g % traj.n_repeat_batch),
        #                                                    **traj.kwargs
        #                                                    )

        best_indviduals = np.argsort(fitness)[::-1]
        current_res = np.sort(fitness)[::-1]
        logger.info('Sorted Fitness {}'.format(current_res))
        self.fitness_all.append(fitness)
        logger.info(
            'Best fitness {} in generation {}'.format(self.current_fitness,
                                                      self.g))
        logger.info('Best individuals index {}'.format(best_indviduals))
        logger.info('Mean of individuals {}'.format(np.mean(current_res)))
        self.best_individual.append((best_indviduals[0], current_res[0]))
        if not self.g % 10 == 0 and self.g >= 0:
            if traj.sample:
                # do sampling only in training phase
                ens, model_outs = self.sample_from_individuals(individuals=ens,
                                                               model_output=model_outs,
                                                               fitness=fitness,
                                                               sampling_method=traj.sampling_method,
                                                               pick_method=traj.pick_method,
                                                               best_n=traj.best_n,
                                                               worst_n=traj.worst_n, # / (self.g % traj.n_repeat_batch /2 + 1),
                                                               **traj.kwargs)
            model_outs = model_outs.reshape((ensemble_size,
                                             len(self.target_label),
                                             len(self.optimizee_labels)))
            enkf = EnKF(maxit=traj.maxit,
                        online=traj.online,
                        n_batches=len(self.optimizee_labels))
            enkf.fit(ensemble=ens,
                     ensemble_size=ensemble_size,
                     observations=np.array(self.optimizee_labels),
                     model_output=model_outs,
                     gamma=gamma)
            # These are all the updated weights for each ensemble
            results = enkf.ensemble.cpu().numpy()
        else: 
            results = ens
            # do the sampling step in test step
            # if traj.sample and not self.g % 10 == 0 and self.g > 0:
        if traj.scale_weights:
            # rescale
            # (np.max(weights) - np.min(weights)) + np.min(weights)
            results = results * np.abs(weights).max()
            # scaler.inverse_transform(results)
        # self.plot_distribution(weights=results, gen=self.g, mean=True)

        if self.g % 1 == 0:
            self.weights_to_save.append(results)

        generation_name = 'generation_{}'.format(traj.generation)
        traj.results.generation_params.f_add_result_group(generation_name)

        generation_result_dict = {
            'generation': traj.generation,
            'connection_weights': results
        }
        traj.results.generation_params.f_add_result(
            generation_name + '.algorithm_params', generation_result_dict)

        # Produce the new generation of individuals
        if traj.stop_criterion <= self.current_fitness or self.g < traj.n_iteration:
            # Create new individual based on the results of the update from the EnKF.
            new_individual_list = [
                {'weights_eeo': results[i][:len(individuals[i].weights_eeo)],
                 'weights_eio': results[i][len(individuals[i].weights_eeo):len(individuals[i].weights_eio) + len(individuals[i].weights_eeo)],
                 'weights_ieo': results[i][len(individuals[i].weights_eeo) + len(individuals[i].weights_eio):len(individuals[i].weights_ieo) + len(individuals[i].weights_eeo) + len(individuals[i].weights_eio)],
                 'weights_iio': results[i][len(individuals[i].weights_eeo) + len(individuals[i].weights_eio) + len(individuals[i].weights_ieo):len(individuals[i].weights_iio) + len(individuals[i].weights_eeo) + len(individuals[i].weights_eio) + len(individuals[i].weights_ieo)],
                 'train_set': self.train_set,
                 'targets': self.optimizee_labels} for i in
                range(ensemble_size)]

            # Check this bounding function
            if self.optimizee_bounding_func is not None:
                new_individual_list = [self.optimizee_bounding_func(ind) for
                                       ind in new_individual_list]

            fitnesses_results.clear()
            self.eval_pop = new_individual_list
            if self.g % traj.n_repeat_batch == 0 and self.g > 0 and not (self.g + 1) % 10 == 0:
                logger.info('Changing dataset in generation {}'.format(self.g))
                if self.data_loader_method == 'random':
                    self.optimizee_labels, self.random_ids = self.randomize_labels(
                        self.train_labels, size=traj.n_batches)
                elif self.data_loader_method == 'separated':
                    self.optimizee_data, self.optimizee_labels = self.get_separated_data(
                        self.train_labels, self.train_set, self.target_label,
                        shuffle=traj.parameters.shuffle,
                        n_slice=traj.parameters.n_slice)
                    if not traj.parameters.shuffle:
                        rand_ind = self.random_state.randint(0, len(self.target_label), 1)[0]
                        self.optimizee_labels, self.random_ids = self.randomize_labels(
                            self.optimizee_labels[rand_ind],
                            size=traj.n_batches)
                logger.info('New targets are {}'.format(self.optimizee_labels))
                for e in self.eval_pop:
                    if self.data_loader_method == 'random':
                        e["targets"] = self.optimizee_labels
                        e["train_set"] = [self.train_set[r] for r in self.random_ids]
                    elif self.data_loader_method == 'separated':
                        if traj.parameters.shuffle:
                            e["targets"] = self.optimizee_labels
                            e["train_set"] = self.optimizee_data
                        else:
                            e["targets"] = self.optimizee_labels
                            e["train_set"] = [self.optimizee_data[rand_ind][r] for r in
                                              self.random_ids]
            # Apply test set
            elif (self.g + 1) % 10 == 0 and self.g > 0:
                logger.info('Testing dataset in generation {}'.format(self.g))
                logger.info('New test targets are {}'.format(self.test_labels[:len(self.optimizee_labels)]))
                for e in self.eval_pop:
                    e["targets"] = [int(t) for t in self.test_labels[:len(self.optimizee_labels)]]
                    e["train_set"] = self.test_set[:len(self.optimizee_labels)]
                self.cov_to_save.append(enkf.cov_mat)

            self.g += 1  # Update generation counter
            self._expand_trajectory(traj)

    @staticmethod
    def _scale_weights(weights, normalize=False, method=pp.MinMaxScaler,
                       **kwargs):
        scaler = 0.
        if normalize:
            scaler = method(**kwargs)
            weights = scaler.fit_transform(weights)
        return weights, scaler

    def sample_from_individuals(self, individuals, fitness, model_output,
                                best_n=0.25, worst_n=0.25,
                                pick_method='random',
                                **kwargs):
        """
        Samples from the best `n` individuals via different methods.

        :param individuals: array_like
            Input data, the individuals
        :param fitness: array_like
            Fitness array
        :param best_n: float
            Percentage of best individuals to sample from
        :param model_output, array like, model outputs of the best indiviudals 
            will be used to replace the model outputs of the worst individuals 
            in the same manner as the sampling
        :param worst_n:
            Percentage of worst individuals to replaced by sampled individuals
        :param pick_method: str
            Either picks the best individual randomly 'random' or it picks the
            iterates through the best individuals and picks with a certain
            probability `best_first` the first best individual
            `best_first`. In the latter case must be used with the key word
            argument `pick_probability`.  `gaussian` creates a multivariate
            normal using the mean and covariance of the best individuals to
            replace the worst individuals.
            Default: 'random'
        :param kwargs:
            'pick_probability': float
                Probability of picking the first best individual. Must be used
                when `pick_method` is set to `pick_probability`.
            'loc': float, mean of the gaussian normal, can be specified if
               `pick_method` is `random` or `pick_probability`
               Default: 0.
            'scale': float, std scale of the gaussian normal, can be specified if
                `pick_method` is `random` or `pick_probability`
                 Default: 0.1
        :return: array_like
            New array of sampled individuals.
        """
        # best fitness should be here ~ 1 (which means correct choice)
        # sort them from best to worst via the index of fitness
        # get indices
        indices = np.argsort(fitness)[::-1]
        sorted_individuals = np.array(individuals)[indices]
        # get best n individuals from the front
        best_individuals = sorted_individuals[:int(len(individuals) * best_n)]
        # get worst n individuals from the back
        worst_individuals = sorted_individuals[
            len(individuals) - int(len(individuals) * worst_n):]
        # sort model outputs
        sorted_model_output = model_output[indices]
        for wi in range(len(worst_individuals)):
            if pick_method == 'random':
                # pick a random number for the best individuals add noise
                rnd_indx = self.random_state.randint(len(best_individuals))
                ind = best_individuals[rnd_indx]
                # add gaussian noise
                noise = self.random_state.normal(loc=kwargs['loc'],
                                                 scale=kwargs['scale'],
                                                 size=len(ind))
                worst_individuals[wi] = ind + noise
                model_output[wi] = sorted_model_output[rnd_indx]
            elif pick_method == 'best_first':
                for bidx, bi in enumerate(best_individuals):
                    pp = kwargs['pick_probability']
                    rnd_pp = self.random_state.rand()
                    if pp >= rnd_pp:
                        # add gaussian noise
                        noise = self.random_state.normal(loc=kwargs['loc'],
                                                         scale=kwargs['scale'],
                                                         size=len(bi))
                        worst_individuals[wi] = bi + noise
                        model_output[wi] = sorted_model_output[bidx]
                        break
            else:
                sampled = self._sample(best_individuals, pick_method)
                worst_individuals = sampled
                rnd_int = self.random_state.randint(
                    0, len(best_individuals), size=len(best_individuals))
                model_output[len(sorted_individuals) -
                             len(worst_individuals):] = sorted_model_output[rnd_int]
                break
        sorted_individuals[len(sorted_individuals) -
                           len(worst_individuals):] = worst_individuals
        return sorted_individuals, model_output

    def _sample(self, individuals, method='gaussian'):
        if method == 'gaussian':
            dist = Gaussian()
            dist.init_random_state(self.random_state)
            dist.fit(individuals)
            sampled = dist.sample(len(individuals))
        elif method == 'rv_histogram':
            sampled = [scipy.stats.rv_histogram(h) for h in individuals]
        else:
            raise KeyError('Sampling method {} not known'.format(method))
        sampled = np.asarray(sampled)
        return sampled

    @staticmethod
    def adjust_similar_lengths(individuals, fitness, bins='auto',
                               method='interpolate'):
        """
        The lengths of the individuals may differ. To fill the individuals to
        the same length sample values from the distribution of the individual
        with the best fitness.
        """
        # check if the sizes are different otherwise skip
        if len(set(
                [len(individuals[i]) for i in range(len(individuals))])) == 1:
            return individuals
        # best fitness should be here ~ 0 (which means correct choice)
        idx = np.argmin(fitness)
        best_ind = individuals[idx]
        best_min = np.min(best_ind)
        best_max = np.max(best_ind)
        hist = np.histogram(best_ind, bins)
        if method == "interpolate":
            # for interpolation it is better to reduce the max and min values
            best_max -= 100
            best_min += 100
            hist_dist = scipy.interpolate.interp1d(hist[1][:-1], hist[0])
        elif method == 'rv_histogram':
            hist_dist = scipy.stats.rv_histogram(hist)
        else:
            raise KeyError('Sampling method {} not known'.format(method))
        # get the longest individual
        longest_ind = individuals[np.argmax([len(ind) for ind in individuals])]
        new_inds = []
        for inds in individuals:
            subs = len(longest_ind) - len(inds)
            if subs > 0:
                rnd = np.random.uniform(best_min, best_max, subs)
                inds.extend(hist_dist(rnd))
                new_inds.append(inds)
            else:
                # only if subs is the longest individual
                new_inds.append(inds)
        return new_inds

    @staticmethod
    def plot_distribution(weights, gen, mean=True):
        """ Plots the weights as a histogram """
        if mean:
            plt.hist(weights.mean(0))
        else:
            plt.hist(weights)
        plt.savefig('weight_distributions_gen{}.pdf'.format(gen), format='pdf')
        plt.close()

    @staticmethod
    def plot_fitnesses(fitnesses):
        std = np.std(fitnesses, axis=1)
        mu = np.mean(fitnesses, axis=1)
        lower_bound = mu - std
        upper_bound = mu + std
        plt.plot(mu, 'o-')
        plt.fill_between(range(len(mu)), lower_bound,
                         upper_bound, alpha=0.5, color='green')
        # plt.plot(np.ones_like(f_) * i, np.ravel(f), '.')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.savefig('fitnesses.pdf', format='pdf')
        plt.savefig('fitnesses.svg', format='svg')
        plt.close()

    @staticmethod
    def _remove_files(suffixes):
        for suffix in suffixes:
            files = glob.glob('*.{}'.format(suffix))
            try:
                [os.remove(fl) for fl in files]
            except OSError as ose:
                print('Error {} {}'.format(files, ose))

    def end(self, traj):
        """
        Run any code required to clean-up, print final individuals etc.
        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`
        """
        traj.f_add_result('final_individual', self.best_individual)
        self.plot_fitnesses(self.fitness_all)
        logger.info(
            "The best individuals with fitness {}".format(
                self.best_individual))
        np.savez_compressed(os.path.join(traj.parameters.path, 'weights.npz'),
                            weights=self.weights_to_save)
        np.savez_compressed(os.path.join(traj.parameters.path, 'fitness_individual.npz'),
                            fitness=self.fitness_all)
        np.savez_compressed(os.path.join(traj.parameters.path, 'covariance_matrices.npz'),
                            covmat=self.cov_to_save)
        logger.info("-- End of (successful) EnKF optimization --")
