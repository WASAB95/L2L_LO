import os
import sys
from datetime import datetime

import pandas as pd
import glob
import numpy as np
import logging
from collections import Counter, namedtuple
from os.path import exists
from sklearn.cluster import DBSCAN

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from l2l.optimizers.NN.MC_Network import Network

from l2l.optimizers.optimizer import Optimizer

from l2l import get_grouped_dict, dict_to_list, list_to_dict

logger = logging.getLogger("optimizers.nn")

NNOptimizerParameters = namedtuple('NNParamters', ['neurons', 'learning_rate', 'epochs', 'batch_size', 'schema',
                                                   'pop_size', 'seed', 'n_iteration', 'stop_criterion', 'input_path',
                                                   'header', 'target_category'])


def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())


def print_weights(model):
    state_dict = model.state_dict()
    logger.info(f"amp layer 1: {state_dict['fa.weight']}")
    logger.info(f"phase layer 1: {state_dict['fp.weight']}")
    logger.info(f"amp layer 2: {state_dict['fao.weight']}")
    logger.info(f"phase layer 2: {state_dict['fpo.weight']}")


def reset_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def cluster_data(df, eps, min_samples):
    clustered_df = df.drop(columns=['category', 'fitness'])
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    print(Counter(clusters.labels_))
    clustered_df['cluster'] = clusters.labels_
    return clustered_df, list(set(clusters.labels_))


def get_clusters_centroids(df, cluster_labels):
    centroids = []
    for l in cluster_labels:
        centroids.append(
            np.array(df[df['cluster'] == l].mean().drop('cluster')))
    return centroids


class NNOptimizer(Optimizer):
    def __init__(self, traj,
                 optimizee_create_individual,
                 optimizee_fitness_weights,
                 parameters,
                 optimizee_bounding_func=None):

        super().__init__(traj,
                         optimizee_create_individual=optimizee_create_individual,
                         optimizee_fitness_weights=optimizee_fitness_weights,
                         parameters=parameters, optimizee_bounding_func=optimizee_bounding_func)
        # Creating Placeholders for individuals and results that are about to be explored
        self.best_individual = {'g': 0, 'f': -np.inf, 'individuals': []}
        self.targets = None
        self.final_df = None
        self.counter = 0
        self.current_centroids = None
        self.centroids_list = []
        self.model = Network().double()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=parameters.learning_rate, momentum=0.9)

        traj.f_add_parameter('learning_rate', parameters.learning_rate, comment='Value of learning rate')
        traj.f_add_parameter('pop_size', parameters.pop_size,
                             comment='Number of minimal individuals simulated in each run')
        traj.f_add_parameter('n_iteration', parameters.n_iteration, comment='Number of iteration to perform')
        traj.f_add_parameter('stop_criterion', parameters.stop_criterion, comment='Stopping criterion parameter')
        traj.f_add_parameter('seed', np.uint32(parameters.seed), comment='Optimizer random seed')

        self.collected_data = []
        self.best_dict = {}
        self.trained = False
        self.training_generations = []

        __, self.optimizee_individual_dict_spec = dict_to_list(optimizee_create_individual(), get_dict_spec=True)

        # current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]
        #
        # if optimizee_bounding_func is not None:
        #     current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        input_data = self.read_data_from_file(path=parameters.input_path,
                                              header=parameters.header)

        self.categorized_df = self.create_categories(input_data, parameters.target_category)

        self.clustered_df, self.clusters_labels = cluster_data(
            self.categorized_df[self.categorized_df['category'] == 1], eps=120,
            min_samples=(len(self.categorized_df.columns) - 2) * 2)

        self.centroids = get_clusters_centroids(self.clustered_df, self.clusters_labels)
        print(self.centroids)

        # target_label = self.clusters_labels[0]
        #
        # import time
        # start_time = time.time()
        # fitness, targets = self.get_train_data(df=self.categorized_df, clustered_df=self.clustered_df,
        #                                        target_label=target_label)
        # print("--- %s seconds ---" % (time.time() - start_time))
        #
        # print("DONE")

        #: The current generation number
        self.g = 0
        self.current_fitness = -np.Inf

        #: The population (i.e. list of individuals) to be evaluated at the next iteration
        self.eval_pop = [list_to_dict(c, self.optimizee_individual_dict_spec)
                         for c in self.centroids]
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        fitnesses_list = [i[1][0] for i in fitnesses_results]

        if self.g == 0:
            cluster_ind = np.argmax(fitnesses_list)
            self.current_centroids = self.centroids[cluster_ind]
            self.centroids_list.append(self.current_centroids)
            target_label = self.clusters_labels[cluster_ind]
            print(fitnesses_list)
            print(target_label)
            if exists('../data/all_data.csv'):
                print('all_data exist')
                df = pd.read_csv('../data/all_data.csv')
                fitness = df['fitness'].values
                targets = df.drop(columns=['fitness']).values
                self.final_df = df
            else:
                fitness, targets = self.get_train_data(df=self.categorized_df, clustered_df=self.clustered_df,
                                                       target_label=target_label)

            train_loader = self.get_train_loader(fitness, targets, self.parameters.batch_size)
            self.train_network(train_loader, self.parameters.epochs)
            current_eval_pop = [self.optimizee_create_individual() for _ in range(self.parameters.pop_size)]
            fitnesses_results.clear()
            self.eval_pop = current_eval_pop
            self.g += 1
            self._expand_trajectory(traj)

        else:

            self.collected_data += fitnesses_list

            weighted_fitness_list = []

            for run_index, fitness in fitnesses_results:
                # We need to convert the current run index into an ind_idx
                # (index of individual within one generation)
                traj.v_idx = run_index
                ind_index = traj.par.ind_idx

                traj.f_add_result('$set.$.individual', old_eval_pop[ind_index])
                traj.f_add_result('$set.$.fitness', fitness)

                weighted_fitness_list.append(np.dot(fitness, self.optimizee_fitness_weights))
            traj.v_idx = -1  # set trajectory back to default

            fitness_sorting_indices = list(reversed(np.argsort(weighted_fitness_list)))
            old_eval_pop_as_array = np.array([dict_to_list(x) for x in old_eval_pop])

            # Sorting the data according to fitness
            sorted_population = old_eval_pop_as_array[fitness_sorting_indices]
            sorted_fitness = np.asarray(weighted_fitness_list)[fitness_sorting_indices]

            logger.info("-- End of generation %d --", self.g)
            logger.info("  Evaluated %d individuals", len(fitnesses_results))
            logger.info('  Average Fitness: %.4f', np.mean(sorted_fitness))
            logger.info("  Current fitness is %.2f", self.current_fitness)
            logger.info('  Best Fitness: %.4f', sorted_fitness[0])
            # logger.info("  Best individual is %s", sorted_population[0])

            self.best_dict[self.g] = sorted_fitness[0]

            if self.best_dict[self.g] > self.best_individual['f']:
                self.best_individual['g'] = self.g
                self.best_individual['f'] = self.best_dict[self.g]
                self.best_individual['individuals'] = sorted_population[0]

            # if self.best_dict[self.g] >= self.parameters.target_category:
            #     self.new_centroids.append(sorted_population[0] - self.current_centroids)

            if self.g > 1 and (self.best_dict[self.g] < self.best_dict[self.g - 1]
                               or abs(self.best_dict[self.g - 1] - self.best_dict[self.g]) <= 0.0001):
                self.counter += 1
            else:
                self.counter = 0

            stop = False

            if 1 < self.g < traj.n_iteration - 1 and self.counter >= 3 and not stop:
                self.model.apply(reset_weight)
                self.centroids_list.append(self.best_individual['individuals'])
                # self.centroids_list = [self.best_individual['individuals']]
                # self.best_individual['f'] = -np.inf
                print(f'centroids_list: {self.centroids_list}')
                adjust = np.mean(self.centroids_list, axis=0) - self.current_centroids
                self.current_centroids = np.mean(self.centroids_list, axis=0)

                print(f'adjust is {adjust}')
                self.final_df[self.targets] = self.final_df[self.targets] + adjust
                # self.final_df.to_csv(f'mapped_data_test_01_{self.g}.csv', index=False)
                targets = self.final_df[self.targets].values
                fitness = self.final_df[['fitness']].values
                train_loader = self.get_train_loader(fitness, targets, self.parameters.batch_size)
                logger.info(f"Training Started {len(self.training_generations)} ...")
                self.train_network(train_loader, self.parameters.epochs)
                self.training_generations.append(self.g)
                self.collected_data.clear()
                self.counter = 0

                if len(self.training_generations) > 3:
                    stop = True

            if self.g < traj.n_iteration - 1:
                new_individuals = [self.model(torch.tensor([f]).double()).tolist() for f in fitnesses_list]
                print(len(new_individuals))
                fitnesses_results.clear()
                self.eval_pop = [list_to_dict(c, self.optimizee_individual_dict_spec)
                                 for c in new_individuals]
                self.g += 1
                self._expand_trajectory(traj)

    def end(self, traj):
        logger.info("  Best individual is %s", self.best_individual['individuals'])
        logger.info('  Best Fitness: %.4f', self.best_individual['f'])
        logger.info("  found in generation %d", self.best_individual['g'])

        # plt.figure(figsize=(self.parameters.n_iteration / 5, self.parameters.n_iteration / 10))
        plt.scatter(self.best_dict.keys(), self.best_dict.values())
        plt.xticks(
            np.arange(min(self.best_dict.keys()), max(self.best_dict.keys()) + 1, self.parameters.n_iteration / 10))
        plt.scatter(self.training_generations, [self.best_dict[i] for i in self.training_generations], color="red")
        plt.xlabel('generation')
        plt.ylabel('best fitness')
        plt.show()
        logger.info(f"Training generations {self.training_generations}")
        """
        Run any code required to clean-up, print final individuals etc.

        :param  ~l2l.utils.trajectory.Trajectory traj: The  trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        """
        pass

    def apply_loss(self, loss_list, epochs):
        for epoch in range(epochs):
            loss = torch.tensor(np.square(loss_list).mean(), requires_grad=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info(f"weights in after adjusting")
        print_weights(self.model)

    def train_network(self, train_loader, epochs):
        loss_list = []
        steps = 0
        for epoch in range(epochs):
            for idx, (data, targets) in enumerate(train_loader):
                data = data.view(data.size(0), 1).to(self.device)
                self.optimizer.zero_grad()

                output = self.model(data)

                loss = self.loss_function(output, targets)
                loss_list.append(loss.item())

                loss.backward()
                self.optimizer.step()

            steps += idx + 1

        step = np.linspace(0, steps, steps)
        plt.plot(step, np.array(loss_list))
        plt.xlabel('training step')
        plt.ylabel('loss')
        plt.show()

        logger.info(f"weights in pre training")
        # print_weights(self.model)

    def get_train_loader(self, fitness, features, batchsize):
        train_tensor = torch.utils.data.TensorDataset(torch.tensor(fitness, requires_grad=True), torch.tensor(features))
        return torch.utils.data.DataLoader(dataset=train_tensor, batch_size=batchsize,
                                           shuffle=True)

    def read_data_from_file(self, path, header):
        all_files = glob.glob(path) if '.csv' in path else glob.glob(os.path.join(path, '*.csv'))
        if header is not None:
            df = pd.concat((pd.read_csv(f, header=header) for f in all_files)).reset_index(
                drop=True)
            self.targets = df.drop(columns=['fitness']).columns.values
        else:
            df = pd.concat(
                (pd.read_csv(f, header=header, names=self.parameters.schema) for f in all_files)).reset_index(
                drop=True)
            self.targets = df.drop(columns=['fitness']).columns.values
        return df

    def create_categories(self, df, threshold):
        df['category'] = pd.cut(df['fitness'], bins=[-np.inf, threshold, 1], include_lowest=True, labels=[0, 1])
        return df

    def get_train_data(self, df, clustered_df, target_label):
        data = df.drop(columns=['category'])

        ref = clustered_df[clustered_df['cluster'] == target_label] \
            .drop(columns=['cluster']).to_numpy()

        # data = df[df['category'] == 0].drop(columns=['category'])
        # ref = df[df['category'] == 1].drop(columns=['category', 'fitness']).to_numpy()

        data[self.targets] = data.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                                        result_type='expand')
        # data.to_csv('mapped_data_mc.csv', index=False)
        self.final_df = data
        return data[['fitness']].values, data[self.targets].values

    # calculate the euclidean distance and return amp,phase of the nearst point
    def min_distance(self, row, ref_array):
        current_individuals = np.array([row[i] for i in self.targets])
        distances = np.array([np.linalg.norm(current_individuals - y) for y in ref_array])
        one_percent = round(len(ref_array) / 100)
        # one_percent = 10
        ind = distances.argsort()[:one_percent]
        # amp = np.mean([a[0] for a in ref_array[ind]])
        # phase = np.mean([p[1] for p in ref_array[ind]])
        targets_list = [np.mean([a[i] for a in ref_array[ind]]) for i in range(len(self.targets))]
        return targets_list
