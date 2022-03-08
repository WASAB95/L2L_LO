import logging
from enum import Enum

import numpy
import matplotlib.pyplot as pl

from l2l.optimizees.optimizee import Optimizee
import ast


logger = logging.getLogger("ltl-sp-simple")


class SPNoVizOptimizee(Optimizee):

    def __init__(self, trajectory, seed):

        super(SPNoVizOptimizee, self).__init__(trajectory)

        seed = numpy.uint32(seed)
        self.random_state = numpy.random.RandomState(seed=seed)

    def init_(self, trajectory):
        '''
        We define general simulation parameters
        '''
        # simulated time (ms)
        self.t_sim = 200000.0
        # simulation step (ms).
        self.dt = 0.1
        self.number_excitatory_neurons = 800
        self.number_inhibitory_neurons = 200
        self.regions = 1

        # Structural_plasticity properties
        self.update_interval = 100
        self.record_interval = 1000.0
        # rate of background Poisson input
        self.bg_rate = 10000.0
        self.neuron_model = 'iaf_psc_alpha'

        '''
        In this implementation of structural plasticity, neurons grow
        connection points called synaptic elements. Synapses can be created
        between compatible synaptic elements. The growth of these elements is
        guided by homeostatic rules, defined as growth curves.
        Here we specify the growth curves for synaptic elements of excitatory
        and inhibitory neurons.
        '''
        # Excitatory synaptic elements of excitatory neurons
        self.growth_curve_e_e = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Hz
            'eps': 5.0,  # Hz
        }

        # Inhibitory synaptic elements of excitatory neurons
        self.growth_curve_e_i = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Ca2+
            'eps': self.growth_curve_e_e['eps'],  # Ca2+
        }

        # Excitatory synaptic elements of inhibitory neurons
        self.growth_curve_i_e = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0004,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Hz
            'eps': 20.0,  # Hz
        }

        # Inhibitory synaptic elements of inhibitory neurons
        self.growth_curve_i_i = {
            'growth_curve': "gaussian",
            'growth_rate': 0.0001,  # (elements/ms)
            'continuous': False,
            'eta': 0.0,  # Hz
            'eps': self.growth_curve_i_e['eps']  # Hz
        }

        '''
        Now we specify the neuron model.
        '''
        self.model_params = {'tau_m': 10.0,  # membrane time constant (ms)
                             # excitatory synaptic time constant (ms)
                             'tau_syn_ex': 0.5,
                             # inhibitory synaptic time constant (ms)
                             'tau_syn_in': 0.5,
                             't_ref': 2.0,  # absolute refractory period (ms)
                             'E_L': -65.0,  # resting membrane potential (mV)
                             'V_th': -50.0,  # spike threshold (mV)
                             'C_m': 250.0,  # membrane capacitance (pF)
                             'V_reset': -65.0  # reset potential (mV)
                             }

        self.nodes_e = [None] * self.regions
        self.nodes_i = [None] * self.regions
        self.loc_e = [[] for i in range(self.regions)]
        self.loc_i = [[] for i in range(self.regions)]
        # Create a list to store mean values
        if nest.Rank() == 0:
            self.mean_fr_e = [[] for i in range(self.regions)]
            self.mean_fr_i = [[] for i in range(self.regions)]
            self.total_connections_e = [[None] * self.regions]  # [[] for i in range(self.regions)]
            self.total_connections_i = [[None] * self.regions]  # [[] for i in range(self.regions)]
            self.last_connections_msg = None
        self.save_state = False
        self.load_state = False

        '''
        We initialize variables for the post-synaptic currents of the
        excitatory, inhibitory and external synapses. These values were
        calculated from a PSP amplitude of 1 for excitatory synapses,
        -1 for inhibitory synapses and 0.11 for external synapses.
        '''
        self.psc_e = 585.0
        self.psc_i = -585.0
        self.psc_ext = 6.2


    def prepare_simulation(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        '''
        We set global kernel parameters. Here we define the resolution
        for the simulation, which is also the time resolution for the update
        of the synaptic elements.
        '''
        nest.SetKernelStatus(
            {
                'resolution': self.dt
            }
        )
        '''
            Set Number of virtual processes. Remember SP does not work well with openMP right now, so calls must always be done using mpiexec
            '''
        nest.SetKernelStatus({'total_num_virtual_procs': self.comm.Get_size()})
        print("Total number of virtual processes set to: " + str(self.comm.Get_size()))

        '''
        Set Structural Plasticity synaptic update interval which is how often
        the connectivity will be updated inside the network. It is important
        to notice that synaptic elements and connections change on different
        time scales.
        '''
        nest.SetStructuralPlasticityStatus({
            'structural_plasticity_update_interval': self.update_interval,
        })

        '''
        Now we define Structural Plasticity synapses. In this example we create
        two synapse models, one for excitatory and one for inhibitory synapses.
        Then we define that excitatory synapses can only be created between a
        pre synaptic element called 'Axon_ex' and a post synaptic element
        called Den_ex. In a similar manner, synaptic elements for inhibitory
        synapses are defined.
        '''
        spsyn_names = ['synapse_in' + str(nam) for nam in range(self.regions)]
        spsyn_names_e = ['synapse_ex' + str(nam) for nam in range(self.regions)]
        sps = {}
        for x in range(0, self.regions):
            nest.CopyModel('static_synapse', 'synapse_in' + str(x))
            nest.SetDefaults('synapse_in' + str(x), {'weight': self.psc_i, 'delay': 1.0})
            nest.CopyModel('static_synapse', 'synapse_ex' + str(x))
            nest.SetDefaults('synapse_ex' + str(x), {'weight': self.psc_e, 'delay': 1.0})
            sps[spsyn_names[x]] = {
                'model': 'synapse_in' + str(x),
                'post_synaptic_element': 'Den_in' + str(x),
                'pre_synaptic_element': 'Axon_in' + str(x),
            }
            sps[spsyn_names_e[x]] = {
                'model': 'synapse_ex' + str(x),
                'post_synaptic_element': 'Den_ex' + str(x),
                'pre_synaptic_element': 'Axon_ex' + str(x),
            }
        nest.SetStructuralPlasticityStatus({'structural_plasticity_synapses': sps})

    def create_nodes(self):
        '''
        Now we assign the growth curves to the corresponding synaptic elements
        '''
        synaptic_elements_e = {}
        synaptic_elements_i = {}

        all_e_nodes = nest.Create('iaf_psc_alpha', self.number_excitatory_neurons * self.regions)
        all_i_nodes = nest.Create('iaf_psc_alpha', self.number_inhibitory_neurons * self.regions)

        for x in range(0, self.regions):
            synaptic_elements_e = {
                'Den_ex' + str(x): self.growth_curve_e_e,
                'Den_in' + str(x): self.growth_curve_e_i,
                'Axon_ex' + str(x): self.growth_curve_e_e,
            }
            synaptic_elements_i = {
                'Den_ex' + str(x): self.growth_curve_i_e,
                'Den_in' + str(x): self.growth_curve_i_i,
                'Axon_in' + str(x): self.growth_curve_i_i,
            }

            self.nodes_e[x] = all_e_nodes[x * self.number_excitatory_neurons:(x + 1) * self.number_excitatory_neurons]
            self.nodes_i[x] = all_i_nodes[x * self.number_inhibitory_neurons:(x + 1) * self.number_inhibitory_neurons]

            self.loc_e[x] = [stat['global_id'] for stat in nest.GetStatus(self.nodes_e[x]) if stat['local']]
            self.loc_i[x] = [stat['global_id'] for stat in nest.GetStatus(self.nodes_i[x]) if stat['local']]
            nest.SetStatus(self.loc_e[x], {'synaptic_elements': synaptic_elements_e})
            nest.SetStatus(self.loc_i[x], {'synaptic_elements': synaptic_elements_i})

    def connect_external_input(self):
        '''
        We create and connect the Poisson generator for external input
        '''
        noise = nest.Create('poisson_generator')
        nest.SetStatus(noise, {"rate": self.bg_rate})
        for x in range(0, self.regions):
            nest.Connect(noise, self.nodes_e[x], 'all_to_all',
                         {'weight': self.psc_ext, 'delay': 1.0})
            nest.Connect(noise, self.nodes_i[x], 'all_to_all',
                         {'weight': self.psc_ext, 'delay': 1.0})

    def get_num_regions(self):
        return self.regions

    def record_ca(self):
        for x in range(0, self.regions):
            fr_e = nest.GetStatus(self.loc_e[x], 'fr'),  # Firing rate
            fr_e = self.comm.gather(fr_e, root=0)
            fr_i = nest.GetStatus(self.loc_i[x], 'fr'),  # Firing rate
            fr_i = self.comm.gather(fr_i, root=0)
            if nest.Rank() == 0:
                mean = numpy.mean(list(fr_e))
                self.mean_fr_e[x].append(mean)
                mean = numpy.mean(list(fr_i))
                self.mean_fr_i[x].append(mean)

    def record_connectivity(self):
        for x in range(0, self.regions):
            syn_elems_e = nest.GetStatus(self.loc_e[x], 'synaptic_elements')
            syn_elems_i = nest.GetStatus(self.loc_i[x], 'synaptic_elements')
            sum_neurons_e = sum(neuron['Axon_ex' + str(x)]['z_connected'] for neuron in syn_elems_e)
            sum_neurons_e = self.comm.gather(sum_neurons_e, root=0)
            sum_neurons_i = sum(neuron['Axon_in' + str(x)]['z_connected'] for neuron in syn_elems_i)
            sum_neurons_i = self.comm.gather(sum_neurons_i, root=0)

            if nest.Rank() == 0:
                self.total_connections_i[x].append (sum(sum_neurons_i))
                self.total_connections_e[x].append (sum(sum_neurons_e))

    def plot_data(self):
        fig, ax1 = pl.subplots()
        ax1.axhline(self.growth_curve_e_e['eps'],
                    linewidth=4.0, color='#9999FF')
        ax1.plot(self.mean_fr_e, 'b',
                 label='Firing rate excitatory neurons', linewidth=2.0)
        ax1.axhline(self.growth_curve_i_e['eps'],
                    linewidth=4.0, color='#FF9999')
        ax1.plot(self.mean_fr_i, 'r',
                 label='Firing rate inhibitory neurons', linewidth=2.0)
        ax1.set_ylim([0, 27.5])
        ax1.set_xlabel("Time in [s]")
        ax1.set_ylabel("Firing rate [Hz]")
        ax1.legend(loc=1)
        pl.savefig('StructuralPlasticityExample.eps', format='eps')

    def simulate_(self):

        nest.Simulate(self.record_interval)

        self.record_ca()
        self.record_connectivity()

    def set_update_interval(self):
        nest.SetStructuralPlasticityStatus({'structural_plasticity_update_interval': self.update_interval, })

    def set_growth_rate(self):
        for x in range(0, self.regions):
            synaptic_elements_e = {'growth_rate': self.egr, }
            synaptic_elements_i = {'growth_rate': self.igr, }
            nest.SetStatus(self.nodes_e[x], 'synaptic_elements_param', {'Den_in' + str(x): synaptic_elements_i})
            nest.SetStatus(self.nodes_e[x], 'synaptic_elements_param', {'Den_ex' + str(x): synaptic_elements_e})
            nest.SetStatus(self.nodes_e[x], 'synaptic_elements_param', {'Axon_ex' + str(x): synaptic_elements_e})
            nest.SetStatus(self.nodes_i[x], 'synaptic_elements_param', {'Axon_in' + str(x): synaptic_elements_e})
            nest.SetStatus(self.nodes_i[x], 'synaptic_elements_param', {'Den_in' + str(x): synaptic_elements_i})
            nest.SetStatus(self.nodes_i[x], 'synaptic_elements_param', {'Den_ex' + str(x): synaptic_elements_e})

    def set_eta(self):
        eta_dict = self.eta
        for x in range(0, self.regions):
            synaptic_elements_e = {'eta': eta_dict[x], }
            nest.SetStatus(self.nodes_e[x], 'synaptic_elements_param', {'Den_in' + str(x): synaptic_elements_e})
            nest.SetStatus(self.nodes_i[x], 'synaptic_elements_param', {'Axon_in' + str(x): synaptic_elements_e})


    def get_fitness(self):
        if(nest.Rank() == 0):
            #firing rate of ex neurons is 80% of the fitness
            fite = (numpy.mean(self.mean_fr_e[-1]) - self.growth_curve_e_e['eps'])/self.growth_curve_e_e['eps']

            #firing rate of the inh neurons is 20% of the fitness
            fiti = (numpy.mean(self.mean_fr_e[-1]) - self.growth_curve_i_i['eps'])/self.growth_curve_i_i['eps']
            fitness = [fite*0.8 + fiti*0.2]
        else:
            fitness = None
        print(str(nest.Rank()) + " / " + str(fitness))
        fitness = self.comm.bcast(fitness,root=0)
        print(str(nest.Rank()) + " / " + str(fitness))
        return fitness

    def simulate(self, trajectory):
        global nest
        import nest
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = self.comm.Get_rank()
        assert (nest.Rank() == self.rank)
        print(self.rank)
        print(trajectory.individual)
        self.id = trajectory.individual.ind_idx
        self.egr = trajectory.individual.egr
        self.igr = trajectory.individual.igr
        self.init_(trajectory)
        self.prepare_simulation()
        self.create_nodes()
        self.connect_external_input()
        self.set_growth_rate()
        nest.EnableStructuralPlasticity()
        for i_counter in range(150):
            self.simulate_()
            if nest.Rank() == 0:
                print('Iteration ' + str(i_counter) + ' finished\n')
        print(self.get_fitness())
        return(self.get_fitness())


    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        self.bound_gr = [0,0]
        self.bound_gr[0] = -0.1#0.0001
        self.bound_gr[1] = 0.1
        return {'egr': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0],
                'igr': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0]}

    def bounding_func(self, individual):
        return individual


