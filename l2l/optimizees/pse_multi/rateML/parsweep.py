import math
import time
import logging
import itertools
import argparse
import os, sys
import pickle

import numpy as np
from numpy import corrcoef

from tvb.simulator.lab import *
# from tvb.rateML.rateML_CUDA import LEMS2CUDA

class TVB_test:

	def __init__(self):
		self.args = self.parse_args()
		self.sim_length = self.args.n_time # 400
		self.g = np.array([1.0])
		self.s = np.array([1.0])
		self.dt = 0.1
		self.period = 1.0
		self.omega = 60.0 * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(self.s, self.g, self.args.tvbn)
		self.integrator = integrators.EulerDeterministic(dt=self.dt)
		self.weights = self.SC = self.connectivity.weights
		self.lengths = self.connectivity.tract_lengths
		self.n_nodes = self.weights.shape[0]
		self.tavg_period = 10.0
		self.nstep = self.args.n_time  # 4s
		self.n_inner_steps = int(self.tavg_period / self.dt)
		self.nc = self.args.n_coupling
		self.ns = self.args.n_speed
		self.couplings, self.speeds = self.setup_params(self.nc, self.ns)
		self.params = self.expand_params(self.couplings, self.speeds)
		self.n_work_items, self.n_params = self.params.shape
		self.min_speed = self.speeds.min()
		self.buf_len_ = ((self.lengths / self.min_speed / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of 2
		self.states = self.args.stts

	def tvb_connectivity(self, speed, global_coupling, tvbnodes):
		white_matter = connectivity.Connectivity.from_file(source_file="connectivity_"+tvbnodes+".zip")
		white_matter.configure()
		white_matter.speed = np.array([speed])
		white_matter_coupling = coupling.Linear(a=global_coupling)
		return white_matter, white_matter_coupling

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')
		parser.add_argument('-c', '--n_coupling', help='num grid points for coupling parameter', default=32, type=int)
		parser.add_argument('-s', '--n_speed', help='num grid points for speed parameter', default=32, type=int)
		parser.add_argument('-t', '--test', help='check results', action='store_true')
		parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=400)
		parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true', default='')
		# parser.add_argument('-p', '--no_progress_bar', help='suppress progress bar', action='store_false')
		parser.add_argument('--caching',
		 					choices=['none', 'shared', 'shared_sync', 'shuffle'],
		 					help="caching strategy for j_node loop (default shuffle)",
		 					default='none'
		 					)
		parser.add_argument('--node_threads', default=1, type=int)
		parser.add_argument('--model',
							#choices=['Rwongwang', 'Kuramoto', 'Epileptor', 'Oscillator', \
							#		 'Oscillatorref', 'Kuramotoref', 'Rwongwangref', 'Epileptorref'],
							help="neural mass model to be used during the simulation",
							default='Oscillator'
							)
		parser.add_argument('--lineinfo', default=True, action='store_true')

		parser.add_argument('--filename', default="kuramoto_network.c", type=str,
							help="Filename to use as GPU kernel definition")

		# parser.add_argument('-b', '--bench', default="regular", type=str, help="What to bench: regular, numba, cuda")

		parser.add_argument('-bx', '--blockszx', default="32", type=int, help="Enter block size x")
		parser.add_argument('-by', '--blockszy', default="32", type=int, help="Enter block size y")

		parser.add_argument('-val', '--validate', default=False, help="Enable validation to refmodels")

		parser.add_argument('--stts', default="1", type=int, help="Number of states of model")
		parser.add_argument('--tvbn', default="68", type=str, help="Number of tvb nodes")

		# for L2L interface. the process ID
		parser.add_argument('--procid', default="0", type=int, help="Number of L2L processes(Only when in L2L)")

		args = parser.parse_args()
		return args

	def expand_params(self, couplings, speeds):  # {{{
		# the params array is transformed into a 2d array
		# by first creating tuples of (speed, coup) and arrayfying then
		# pycuda (check) threats them as flattenened arrays but numba needs 2d indexing

		params = np.array([vals for vals in zip(speeds, couplings)], np.float32)
		print('params', params)
		return params  # }}}

	def setup_params(self, nc, ns):  # {{{
		# Reading the parameters from file dumped by L2L
		# set correct folder if necessary
		coupling_file = open('rateML/couplings_%d' % self.args.procid, 'rb')
		couplings = pickle.load(coupling_file)
		coupling_file.close()

		speed_file = open('rateML/speeds_%d' % self.args.procid, 'rb')
		speeds = pickle.load(speed_file)
		speed_file.close()

		return np.array(couplings), np.array(speeds)  # }}}

	def start_cuda(self, logger):
		# logger.info('start Cuda run')
		from parsweep_cuda import Parsweep
		cudarun = Parsweep()
		tavg_data, corr = cudarun.run_simulation(self.weights, self.lengths, self.params, self.speeds, logger,
										   self.args, self.n_nodes, self.n_work_items, self.n_params, self.nstep,
										   self.n_inner_steps, self.buf_len, self.states, self.dt, self.min_speed)

		return tavg_data, corr

	def set_CUDAmodel_dir(self):
		# self.args.filename = os.path.join(os.path.join(os.getcwd(), os.pardir), 'cuda_refs', self.args.model.lower() + '.c')
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))), 
								 self.args.model.lower() + '.c')

	def calc_corrcoef(self, corr):
		# calculate correlation between SC and simulated FC. SC is the weights of TVB simulation.
		SC = self.connectivity.weights / self.connectivity.weights.max()
		ccFCSC = np.zeros(self.nc*self.ns, 'f')
		for i in range(self.nc*self.ns):
			ccFCSC[i] = np.corrcoef(corr[:, :, i].ravel(), SC.ravel())[0, 1]

		return ccFCSC

	def startsim(self):

		tic = time.time()
		logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
		logger = logging.getLogger('[TVB_CUDA]')

		self.set_CUDAmodel_dir()

		tac = time.time()
		# logger.info("Setup in: {}".format(tac - tic))

		tavg0, corr = self.start_cuda(logger)
		toc = time.time()

		if (self.args.validate==True):
			self.compare_with_ref(logger, tavg0)

		toc = time.time()
		elapsed = toc - tic

		# write output to file
		tavg_file = open('tavg_data', 'wb')
		pickle.dump(tavg0, tavg_file)
		tavg_file.close()


		corr_file = open('corr', 'wb')
		pickle.dump(corr, corr_file)
		corr_file.close()

		print('corr', corr.shape)
		ccFCSC = self.calc_corrcoef(corr)
		print('ccFCSC', ccFCSC.shape)
                
		resL2L_file = open('rateML/result_%d' % self.args.procid, 'wb')
		pickle.dump(self.calc_corrcoef(corr), resL2L_file)
		resL2L_file.close()

		print('Finished CUDA simulation successfully in: {0:.3f}'.format(elapsed))
		print('in {0:.3f} M step/s'.format(1e-6 * self.nstep * self.n_inner_steps * self.n_work_items / elapsed))
		# logger.info('finished')


if __name__ == '__main__':

	np.random.seed(79)

	example = TVB_test()

	# start templating the model specified on cli
	#here = os.path.abspath(os.path.dirname(__file__))
	#LEMS2CUDA.cuda_templating(example.args.model, os.path.join(here, '..', 'XMLmodels'))

	# start simulation with templated model
	example.startsim()
