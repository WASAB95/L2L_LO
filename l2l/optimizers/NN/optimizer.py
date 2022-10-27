from datetime import datetime

import pandas as pd
import glob
import numpy as np
import logging
from collections import namedtuple
from os.path import exists

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from l2l.optimizers.NN.Network import Network
from l2l.optimizers.optimizer import Optimizer
from l2l.utils.tools import cartesian_product

from l2l import get_grouped_dict, dict_to_list

logger = logging.getLogger("optimizers.nn")

NNOptimizerParameters = namedtuple('NNParamters', ['neurons', 'learning_rate', 'epochs', 'batch_size', 'schema',
                                                   'pop_size', 'seed', 'n_iteration', 'stop_criterion', 'input_path',
                                                   'header'])


def print_weights(model):
    state_dict = model.state_dict()
    logger.info(f"amp layer 1: {state_dict['fa.weight']}")
    logger.info(f"phase layer 1: {state_dict['fp.weight']}")
    logger.info(f"amp layer 2: {state_dict['fao.weight']}")
    logger.info(f"phase layer 2: {state_dict['fpo.weight']}")


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
        self.model = Network(parameters.neurons).double()
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

        current_eval_pop = [self.optimizee_create_individual() for _ in range(parameters.pop_size)]

        if optimizee_bounding_func is not None:
            current_eval_pop = [self.optimizee_bounding_func(ind) for ind in current_eval_pop]

        if exists('../data_category.csv'):
            print('data category exist')
            df = pd.read_csv('../data_category.csv')
            fitness = df['fitness'].values
            targets = df.drop(columns=['fitness']).values
            test_df = df.sample(1000, random_state=423)
            self.test_fit = test_df['fitness'].values
            self.test_tar = test_df.drop(columns=['fitness']).values
        else:
            input_data = self.read_data_from_file(path=parameters.input_path,
                                                  header=parameters.header)
            df = self.create_categories(input_data, 0.9)
            fitness, targets = self.get_train_data(df)

        train_loader = self.get_train_loader(fitness, targets, parameters.batch_size)
        start = datetime.now().time()
        self.train_network(train_loader, parameters.epochs)
        end = datetime.now().time()
        logger.info(
            f"training time= {datetime.combine(datetime(1, 1, 1), end) - datetime.combine(datetime(1, 1, 1), start)}")
        #: The current generation number
        self.g = 0
        self.current_fitness = -np.Inf

        #: The population (i.e. list of individuals) to be evaluated at the next iteration
        self.eval_pop = current_eval_pop
        self._expand_trajectory(traj)

    def post_process(self, traj, fitnesses_results):
        old_eval_pop = self.eval_pop.copy()
        self.eval_pop.clear()

        logger.info("  Evaluating %i individuals" % len(fitnesses_results))

        # df = pd.DataFrame(old_eval_pop)
        fitnesses_list = [i[1][0] for i in fitnesses_results]
        # df['fitness'] = fitnesses_list
        self.collected_data += fitnesses_list
        # self.collected_data = pd.concat([self.collected_data, df], ignore_index=True)

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
        logger.info("  Best individual is %s", sorted_population[0])

        self.best_dict[self.g] = sorted_fitness[0]

        stop = False
        if 0 < self.g < traj.n_iteration - 1 and abs(self.best_dict[self.g - 1] - self.best_dict[self.g]) <= 0.0001:
            if self.trained:
                stop = True
            else:
                # compute the difference between current and previous fitness and set a threshold to trigger this
                logger.info(f"Training  STARTED in generation {self.g}")
                collected_fitness = [pow(i, len(self.training_generations) + 1) for i in self.collected_data]
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.1
                self.apply_loss(collected_fitness, epochs=100)
                self.trained = True
                self.training_generations.append(self.g)
                # self.collected_data.clear()
        else:
            self.trained = False

        if self.g < traj.n_iteration - 1 and not stop:
            new_individuals = [self.model(torch.tensor([f]).double().view(1, 1)) for f in fitnesses_list]
            new_individual_list = []
            for r in new_individuals:
                temp = {}
                for k in r.keys():
                    temp[k] = r[k].item()
                new_individual_list.append(temp)

            fitnesses_results.clear()
            self.eval_pop = new_individual_list
            self.g += 1
            self._expand_trajectory(traj)

    def end(self, traj):
        plt.figure(figsize=(self.parameters.n_iteration / 5, self.parameters.n_iteration / 10))
        plt.scatter(self.best_dict.keys(), self.best_dict.values())
        plt.xticks(
            np.arange(min(self.best_dict.keys()), max(self.best_dict.keys()) + 1, self.parameters.n_iteration / 10))
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

                amp_target, ph_target = torch.split(targets, 1, 1)
                amp_target = amp_target.to(self.device)
                ph_target = ph_target.to(self.device)

                output = self.model(data)
                amp_out = output['amp']
                phase_out = output['phase']

                loss = self.loss_function(amp_out, amp_target)
                loss = loss + self.loss_function(phase_out, ph_target)
                loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            steps += idx + 1

        step = np.linspace(0, steps, steps)
        plt.plot(step, np.array(loss_list))
        plt.xlabel('training step')
        plt.ylabel('loss')
        plt.show()

        logger.info(f"weights in pre training")
        print_weights(self.model)

    def get_train_loader(self, fitness, features, batchsize):
        train_tensor = torch.utils.data.TensorDataset(torch.tensor(fitness, requires_grad=True), torch.tensor(features))
        return torch.utils.data.DataLoader(dataset=train_tensor, batch_size=batchsize,
                                           shuffle=True)

    def read_data_from_file(self, path, header):
        all_files = glob.glob(path)
        if self.parameters.header is not None:
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

    def get_train_data(self, df):
        data = df[df['category'] == 0].drop(columns=['category'])
        ref = df[df['category'] == 1].drop(columns=['category', 'fitness']).to_numpy()

        data[self.targets] = data.apply(lambda row: self.min_distance(row=row, ref_array=ref), axis=1,
                                        result_type='expand')
        return data[['fitness']].values, data[self.targets].values

    # calculate the euclidean distance and return amp,phase of the nearst point
    def min_distance(self, row, ref_array):
        current_individuals = np.array([row[i] for i in self.targets])
        distances = [np.linalg.norm(current_individuals - y) for y in ref_array]
        ind = np.argmin(distances)
        return ref_array[ind][0], ref_array[ind][1]
