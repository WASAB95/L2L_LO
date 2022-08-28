from collections import namedtuple
import numpy as np
from l2l.optimizees.optimizee import Optimizee

SignalGeneratorOptimizeeParameters = namedtuple('SignalGeneratorOptimizeeParameters', [
    'frequency',
    'amplitude',
    'phase',
    'seed',
    'range'])


class SignalGeneratorOptimizee(Optimizee):
    '''
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations that
    interact with the environment. Given a set of parameters, it runs the simulation and returns the fitness achieved
    with those parameters.
    '''

    def __init__(self=None, traj=None, parameters=None):
        super().__init__(traj)
        traj.f_add_parameter_group('individual', 'Contains parameters of the optimizee')
        self.random = np.random.RandomState(parameters.seed)
        self.fr = parameters.frequency
        self.range = parameters.range
        self.am = [self.random.uniform(parameters.amplitude[0], parameters.amplitude[1]) for i in range(self.range)]
        self.ph = [self.random.uniform(parameters.phase[0], parameters.phase[1]) for i in range(self.range)]

    def create_individual(self):
        amp = self.random.normal(1, 0.5)
        phase = self.random.normal(1, 0.5)
        return dict(amp=amp, phase=phase)

    def bounding_func(self, individual):
        return {
            'amp': np.clip(-2, 2, individual['amp']),
            'phase': np.clip(-1, 1, individual['phase'])}

    def simulate(self, traj):
        amp = traj.individual.amp
        phase = traj.individual.phase
        y_pred = [self.generate_signal(amp, phase, self.fr)]
        y_real = [self.generate_signal(self.am[i], self.ph[i], self.fr) for i in range(self.range)]
        y_mean = np.mean(y_real, axis=0)

        # return self.mean_square_erorr_list(y_pred, y_real)
        return self.mean_square_erorr(y_pred=y_pred, y_true=y_mean)

    def generate_signal(self, amplitude, phase, freq, time=np.arange(0, 1, 0.01)):
        return amplitude * np.sin(2 * np.pi * freq * time + phase)

    def mean_square_erorr(self, y_pred, y_true):
        print(y_true)
        squared_error = (y_true - y_pred) ** 2
        sum_squared_error = np.sum(squared_error)
        loss = sum_squared_error / y_true.size
        print(loss)
        return (1 - loss,)

    def mean_square_erorr_list(self, y_pred, y_true):
        print(y_true)
        print(y_pred)
        squared_error = np.subtract(y_true, y_pred) ** 2
        loss = [np.sum(i) / len(y_pred) for i in squared_error]
        print('los is: ')
        print(loss)
        avg_loss = np.mean(loss)
        print('avg los is: ' + str(avg_loss))
        return (1 - avg_loss,)
