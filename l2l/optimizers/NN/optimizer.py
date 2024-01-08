import os
import sys
from datetime import datetime
import time

import pandas as pd
import glob
import numpy as np
import logging
from collections import Counter, namedtuple
from os.path import exists

from matplotlib.ticker import MaxNLocator
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


def normalize_df(df, min=-20, max=20):
    return (df - min) / (max - min)


def denormalize_df(df, min=-20, max=20):
    return df * (max - min) + min


def sample_normal(seed, x, num_samples, std):
    np.random.seed(seed)
    samples = np.array([np.random.normal(i, std, num_samples) for i in x])
    paired_tuples = samples.T
    return paired_tuples


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


def reset_dictionary(keys):
    dic = {'fitness': []}
    for k in keys:
        dic[k] = []
    return dic


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
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=parameters.learning_rate, momentum=0.2)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=parameters.learning_rate)

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
        self.shift = False
        self.approach = ''

        __, self.optimizee_individual_dict_spec = dict_to_list(optimizee_create_individual(), get_dict_spec=True)

        # current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]
        #
        # if optimizee_bounding_func is not None:
        #     current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        input_data = self.read_data_from_file(path=parameters.input_path,
                                              header=parameters.header)

        input_data[self.targets] = normalize_df(input_data[self.targets])

        self.categorized_df = self.create_categories(input_data, parameters.target_category)

        self.clustered_df, self.clusters_labels = cluster_data(
            self.categorized_df[self.categorized_df['category'] == 1], eps=120,
            min_samples=(len(self.categorized_df.columns) - 2) * 2)

        self.centroids = get_clusters_centroids(self.clustered_df, self.clusters_labels)
        print(self.centroids)

        # self.collected_data = reset_dictionary(self.targets)

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
            logger.info(f"centroid list {self.centroids_list}")
            target_label = self.clusters_labels[cluster_ind]
            print(fitnesses_list)
            print(target_label)
            if exists('../data/mc_mapped_00000.csv'):
                print('mc_mapped_00.csv exist')
                df = pd.read_csv('../data/mc_mapped_00.csv')
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

            # self.collected_data += fitnesses_list
            for ind, p in enumerate(old_eval_pop):
                self.collected_data.append(np.append(p['weights'], fitnesses_list[ind]))

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
            # logger.info("  Current fitness is %.2f", self.current_fitness)
            logger.info('  Best Fitness: %.4f', sorted_fitness[0])
            # logger.info("  Best individual is %s", sorted_population[0])

            if np.isnan(sorted_fitness[0]):
                self.best_dict[self.g] = self.best_dict[self.g - 1]
            else:
                self.best_dict[self.g] = sorted_fitness

            if self.best_dict[self.g][0] - self.best_individual['f'] >= 0.01:
                self.best_individual['g'] = self.g
                self.best_individual['f'] = self.best_dict[self.g][0]
                self.best_individual['individuals'] = sorted_population[0]
                self.counter = 0

            # self.save_n_generation = 10
            # if self.g % self.save_n_generation == 0:
            #     df = pd.DataFrame({'ind': sorted_population[0]})
            #     df.T.to_csv(f"/home/wasab/L2L_V2/collected_individuals/individuals_gen_{self.g}.csv", index=False, header=False,)

            if self.g > 1 and (self.best_dict[self.g][0] < self.best_dict[self.g - 1][0]
                               or abs(self.best_dict[self.g - 1][0] - self.best_dict[self.g][0]) <= 0.01):
                self.counter += 0.5

            elif self.best_dict[self.g][0] < 0 < self.best_individual['f']:
                self.counter += 0.75

            else:
                self.counter = self.counter - 0.5 if self.counter >= 0.5 else 0

            stop = False

            if self.best_individual['f'] >= self.parameters.target_category:
                if self.g > 1 and (self.best_dict[self.g][0] < self.best_dict[self.g - 1][0]
                                   or abs(self.best_dict[self.g - 1][0] - self.best_dict[self.g][0]) <= 0.0001):
                    self.counter += 0.25

            logger.info(f"counter is: {self.counter}")

            count_limit = 2 if self.g >= traj.n_iteration / 2 else 2
            if 1 < self.g < traj.n_iteration - 1 and self.counter >= count_limit and not stop:
                #                 self.model.apply(reset_weight)
                #
                #                 # if len(self.training_generations) == 0:
                #                 #     self.centroids_list = [self.best_individual['individuals']]
                #                 # else:
                #                 #     self.centroids_list.append(self.best_individual['individuals'])
                #
                #
                #                 # self.centroids_list.append(self.best_individual['individuals'])
                #                 # self.centroids_list = [self.best_individual['individuals']]
                #
                #                 # shiftting approach
                #
                #                 collected_data_df = pd.DataFrame(self.collected_data, columns=np.append(self.targets, 'fitness'))
                #                 collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])
                #                 #
                #                 filter_collect = collected_data_df[collected_data_df['fitness'] > self.parameters.target_category]
                #                 filter_final = self.final_df[self.final_df['fitness'] > self.parameters.target_category]
                #                 #
                #                 merged_df = pd.concat([filter_collect, filter_final], ignore_index=True)
                #                 #
                #                 num = 5
                #                 precent = num if (int(len(merged_df) * (num / 100)) > 0) else 100
                #
                #
                # ########
                #                 merged_df = merged_df.sort_values(by=['fitness'], ascending=False).drop_duplicates().head(
                #                     int(len(merged_df) * (precent / 100)))
                #
                #                 logger.info(f"top {precent}% fitness: {merged_df[['fitness']].values}")
                #                 # ref = np.concatenate((filter_collect.drop(columns=['fitness']).to_numpy(), merged_df.drop(columns=['fitness']).to_numpy()))
                #                 ref = merged_df.drop(columns=['fitness']).to_numpy()
                # ########
                #
                #                 # if len(filter_collect) > 0:
                #                 #     ref = filter_collect.drop(columns=['fitness']).to_numpy()
                #                 #     logger.info(f"collected fitness: {filter_collect[['fitness']].values}")
                #                 # else:
                #                 #     ref = merged_df.drop(columns=['fitness']).to_numpy()
                #                 #     logger.info(f"top {precent}% fitness: {merged_df[['fitness']].values}")
                #
                #
                #                 collected_data_df[self.targets] = collected_data_df.apply(
                #                     lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #                     result_type='expand')
                #
                #                 # self.centroids_list = [self.best_individual['individuals']]
                #                 self.centroids_list.append(self.best_individual['individuals'])
                #                 logger.info(f"centroids fitness {self.best_individual['f']}")
                #                 logger.info(f"number of centroids {len(self.centroids_list)}")
                #                 adjust = np.mean(self.centroids_list, axis=0) - self.current_centroids
                #                 self.current_centroids = np.mean(self.centroids_list, axis=0)
                #                 #
                #                 # # print(f'adjust is {adjust}')
                #                 self.final_df[self.targets] = self.final_df[self.targets] + adjust
                #                 ### self.final_df.to_csv(f'mapped_data_test_01_{self.g}.csv', index=False)
                #                 #
                #                 self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #

                # ### Hybrid Approach ###
                # precent = 10
                # logger.info("Shiftting Started")
                # #
                # if not self.shift:
                #     # self.model.apply(reset_weight)
                #
                #     collected_data_df = pd.DataFrame(self.collected_data, columns=np.append(self.targets, 'fitness'))
                #     # collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])
                #     print(collected_data_df)
                #
                #     filter_collect = collected_data_df[collected_data_df['fitness'] > self.parameters.target_category]
                #     filter_final = self.final_df[self.final_df['fitness'] > self.parameters.target_category]
                #
                #
                #     merged_df = pd.concat([filter_collect, filter_final], ignore_index=True)
                #
                #     merged_df = merged_df.sort_values(by=['fitness'], ascending=False).drop_duplicates().head(int(len(merged_df) * (precent / 100)))
                #     logger.info(f"top {precent}% fitness: {merged_df[['fitness']].values}")
                #
                #     ref = merged_df.drop(columns=['fitness']).to_numpy()
                #
                #     collected_data_df[self.targets] = collected_data_df.apply(
                #         lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #         result_type='expand')
                #
                #
                #     # if len(self.training_generations) == 0:
                #     #     self.centroids_list = [self.best_individual['individuals']]
                #     # else:
                #     #     self.centroids_list.append(self.best_individual['individuals'])
                #
                #     # self.centroids_list = [self.best_individual['individuals']]
                #     self.centroids_list.append(self.best_individual['individuals'])
                #     adjust = np.mean(self.centroids_list, axis=0) - self.current_centroids
                #     self.current_centroids = np.mean(self.centroids_list, axis=0)
                #
                #     # print(f'adjust is {adjust}')
                #     self.final_df[self.targets] = self.final_df[self.targets] + adjust
                #     # self.final_df.to_csv(f'mapped_data_test_01_{self.g}.csv', index=False)
                #
                #
                #
                #     self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #
                #     self.shift =True
                #
                # else:
                #     logger.info("Mapping Started")
                #
                #     collected_data_df = pd.DataFrame(self.collected_data, columns=np.append(self.targets, 'fitness'))
                #     # collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])
                #     self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #
                #     filter_new_df = self.final_df[self.final_df['fitness'] > self.parameters.target_category]
                #     # ref = filter_new_df.sort_values(by=['fitness'],ascending=False).head(int(len(filter_new_df)*(precent/100))).drop(columns=['fitness']).to_numpy()
                #
                #     ref = filter_new_df.sort_values(by=['fitness'],ascending=False).drop_duplicates().head(int(len(filter_new_df)*(precent/100)))
                #     logger.info(f"top {precent}% fitness: {ref[['fitness']].values}")
                #     ref = ref.drop(columns=['fitness']).to_numpy()
                #
                #     self.final_df[self.targets] = self.final_df.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #                                     result_type='expand')
                #     self.shift = False

                ##### re-mapping approahc ##########
                #
                # collected_data_df = pd.DataFrame(self.collected_data, columns=np.append(self.targets, 'fitness'))
                # # collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])
                # self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #
                # filter_new_df = self.final_df[self.final_df['fitness'] > self.parameters.target_category]
                #
                # ref = filter_new_df.sort_values(by=['fitness'],ascending=False).drop_duplicates().head(int(len(filter_new_df)*(precent/100)))
                # logger.info(f"top {precent}% fitness: {ref[['fitness']].values}")
                # ref = ref.drop(columns=['fitness']).to_numpy()
                #
                # self.final_df[self.targets] = self.final_df.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #                                 result_type='expand')

                ################# Re-sampling Approach #####################

                # collected_data_df = pd.DataFrame(self.collected_data, columns=np.append(self.targets, 'fitness'))
                # collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])
                # self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #
                # filter_new_df = self.final_df[self.final_df['fitness'] > self.parameters.target_category]
                #
                # precent = 10
                # top_5 = collected_data_df[collected_data_df['fitness'] > self.parameters.target_category]
                # top_5 = collected_data_df.sort_values(by=['fitness'],ascending=False).head(int(len(filter_new_df)*(precent/100)))
                # logger.info(f"top {precent}% fitness: {top_5[['fitness']].values}")
                # top_5 = top_5.drop(columns=['fitness'])
                #
                # total_samples = []
                # for col in top_5.columns.values:
                #     mean = np.mean(top_5[col])
                #     samples = np.random.normal(mean, 0.05, len(filter_new_df))
                #     total_samples.append(samples)
                #
                # ref = np.array([list(i) for i in zip(*total_samples)])
                #
                # self.final_df[self.targets] = self.final_df.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #                                 result_type='expand')
                #
                # logger.info(f"final_df length {len(self.final_df)}")

                ########################################################################

                ##################################
                # MAP ALL to SAMPLE
                self.approach = 'map_all_sample_mix'

                self.model.apply(reset_weight)

                collected_data_df = pd.DataFrame(self.collected_data,
                                                 columns=np.append(self.targets, 'fitness')).drop_duplicates().dropna()

                collected_data_df[self.targets] = normalize_df(collected_data_df[self.targets])

                target_individuals = collected_data_df[self.targets][
                    collected_data_df['fitness'] >= self.parameters.target_category].drop_duplicates().to_numpy()

                if np.any(target_individuals):
                    best_ind = np.mean(target_individuals, axis=0)
                else:
                    best_ind = normalize_df(pd.DataFrame(self.best_individual['individuals'])).to_numpy().flatten()

                std = 0.1

                if len(self.training_generations) < 1:

                    num_samples = int(len(self.final_df) * 0.3)

                    logger.info(f"Resample {num_samples} samples from the best only ")
                    ref = sample_normal(seed=self.g, std=std, x=best_ind,
                                        num_samples=num_samples)

                    self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)

                    self.final_df[self.targets] = self.final_df.apply(
                        lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                        result_type='expand')

                    temp_df = self.final_df.copy()

                    self.centroids_list = [np.mean(ref, axis=0)]
                    self.current_centroids = np.mean(ref, axis=0)
                    logger.info(f"current centroids {self.current_centroids}")

                else:

                    num_samples = int(len(collected_data_df) * 3)

                    logger.info(f"Resample {num_samples} samples from the best only ")
                    ref = sample_normal(seed=self.g, std=std, x=best_ind,
                                        num_samples=num_samples)

                    collected_data_df[self.targets] = collected_data_df.apply(
                        lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                        result_type='expand')


                    logger.info(f"self.best_individual['individuals'] {best_ind}")

                    self.centroids_list.append(best_ind)

                    logger.info(f"centroids fitness {self.best_individual['f']}")
                    logger.info(f"number of centroids {len(self.centroids_list)}")
                    logger.info(f"current centroids {self.current_centroids}")
                    # logger.info(f"centroid list: {self.centroids_list}")
                    adjust = np.mean(self.centroids_list, axis=0) - self.current_centroids
                    self.current_centroids = np.mean(self.centroids_list, axis=0)
                    # logger.info(f"new centroids: {self.current_centroids}")

                    self.final_df[self.targets] = self.final_df[self.targets] + adjust
                    # logger.info(f"adjust is {adjust}")
                    # logger.info(f"TEST TEST {self.g} !!")
                    # logger.info(f"{self.final_df[self.targets].loc[0].values}")
                    #
                    temp_df = self.final_df.copy()
                    self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)

                ########

                # self.approach = 'map_all_sample'
                #
                # self.model.apply(reset_weight)
                #
                # collected_data_df = pd.DataFrame(self.collected_data,
                #                                  columns=np.append(self.targets, 'fitness')).drop_duplicates().dropna()
                #
                # std = 0.2
                #
                # num_samples = int(len(self.final_df)*0.3)
                #
                # logger.info(f"Resample {num_samples} samples from the best only ")
                # ref = sample_normal(seed=self.g, std=std, x=self.best_individual['individuals'],
                #                     num_samples=num_samples)
                #
                # temp_df = self.final_df.copy()
                # self.final_df = pd.concat([self.final_df, collected_data_df], ignore_index=True)
                #
                #
                # self.final_df[self.targets] = self.final_df.apply(
                #     lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                #     result_type='expand')
                #
                # self.centroids_list = [self.best_individual['individuals']]

                #################################

                targets = self.final_df[self.targets].values
                fitness = self.final_df[['fitness']].values
                train_loader = self.get_train_loader(fitness, targets, self.parameters.batch_size)
                logger.info(f"Training Started {len(self.training_generations)} ...")
                self.train_network(train_loader, self.parameters.epochs)
                self.training_generations.append(self.g)
                # self.collected_data.clear()
                self.counter = 0
                self.final_df = temp_df

                # self.collected_data = reset_dictionary(self.targets)

                if len(self.training_generations) > 3:
                    stop = True

            if self.g < traj.n_iteration - 1:
                new_individuals = [self.model(torch.tensor([f]).double()).tolist() for f in fitnesses_list]
                new_individuals = [list(map(denormalize_df, i)) for i in new_individuals]
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

        df_out = pd.DataFrame({'ind': self.best_individual['individuals']})
        df_out = df_out.T
        df_out.to_csv(
            '/home/wasab/final_results/MC/best_individuals/test_{}'.format(
                datetime.now().strftime("%Y-%m-%d-%H_%M_%S")) + '.csv',
            header=False, index=False)
        # pd.DataFrame(self.best_dict).T.to_csv("/home/wasab/final_results/MC/all_individuals_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")) + '.csv')
        pd.DataFrame(self.best_dict).T.to_csv(
            f'/home/wasab/final_results/MC/{self.best_individual["f"]}_{self.best_individual["g"]}_all_individuals_{format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))}.csv')

        mu = []
        sigma = []
        maxes = []
        for k, v in self.best_dict.items():
            mu.append(np.mean(v, 0))
            sigma.append(np.std(v, 0))
            maxes.append(np.max(v, 0))
        # plt.errorbar(ordered_dict.keys(), mu, sigma, fmt='-o')
        plt.figure(figsize=(10, 8))
        plt.plot(self.best_dict.keys(), mu, '-*')
        plt.plot(self.best_dict.keys(), maxes, '.-', c='g')
        # plt.xticks(np.arange(1, self.parameters.n_iteration, 1))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        y1 = np.array(mu) + np.array(sigma)
        y2 = np.array(mu) - np.array(sigma)
        plt.fill_between(self.best_dict.keys(), y1, y2, alpha=.5)
        plt.scatter(self.training_generations, [mu[i - 1] for i in self.training_generations], color='r',
                    label='Optimizer Training', s=50)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        avg_fit = np.mean([i[0] for i in self.best_dict.values()])
        plt.savefig(
            f'/home/wasab/final_results/MC/fitness_{self.parameters.learning_rate}_{self.parameters.epochs}_{self.parameters.pop_size}_{self.parameters.n_iteration}_{self.approach}_{avg_fit}_{self.best_individual["f"]}_{self.best_individual["g"]}_{time.time()}.pdf',
            bbox_inches='tight', pad_inches=0.1)
        plt.show()

        # plt.figure(figsize=(self.parameters.n_iteration / 5, self.parameters.n_iteration / 10))
        # plt.scatter(self.best_dict.keys(), self.best_dict.values())
        # plt.xticks(
        #     np.arange(min(self.best_dict.keys()), max(self.best_dict.keys()) + 1, self.parameters.n_iteration / 10))
        # plt.scatter(self.training_generations, [self.best_dict[i] for i in self.training_generations], color="red")
        # plt.xlabel('generation')
        # plt.ylabel('best fitness')
        # plt.show()
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

    ## df: data frame with category [1,0] [w0,..,w314,fit,cat]
    ## clusterd_df : only target data with cluster [w0,..,w314,cluster]
    def get_train_data(self, df, clustered_df, target_label):
        data = df.drop(columns=['category'])

        ref = clustered_df[clustered_df['cluster'] == target_label] \
            .drop(columns=['cluster']).to_numpy()

        data[self.targets] = data.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                                        result_type='expand')
        self.final_df = data
        return data[['fitness']].values, data[self.targets].values

    # calculate the euclidean distance and return amp,phase of the nearst point
    def min_distance(self, row, ref_array):
        current_individuals = np.array([row[i] for i in self.targets])
        distances = np.linalg.norm(ref_array - current_individuals, axis=1)
        percent = 1
        x_percent = round(len(ref_array) * percent / 100) if round(len(ref_array) * percent / 100) > 1 else 10
        x_percent = 5
        ind = np.argpartition(distances, x_percent)[:x_percent]

        targets_list = [np.mean(ref_array[ind][:, i]) for i in range(len(self.targets))]

        return targets_list
