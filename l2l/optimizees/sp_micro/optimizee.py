import logging

import pickle
import matplotlib
import numpy

from l2l.optimizees.optimizee import Optimizee

matplotlib.use('Agg')
import matplotlib.pyplot as pl
import sys
logger = logging.getLogger("l2l-sp-micro")
import time

class SPMicrocircuitOptimizee(Optimizee):

    def __init__(self, trajectory, seed):

        super(SPMicrocircuitOptimizee, self).__init__(trajectory)
        self.seed = numpy.uint32(seed)
        self.random_state = numpy.random.RandomState(seed=self.seed)


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
        self.regions = 4

        # Structural_plasticity properties
        self.update_interval = 1000
        self.record_interval = 1000.0
        # rate of background Poisson input
        self.bg_rate = 10000.0
        self.neuron_model = 'iaf_psc_exp'
        self.time = []

        ######################################
        ##         Network parameters      ###
        ######################################

        # area of network in mm^2; scales numbers of neurons
        # use 1 for the full-size network (77,169 neurons)
        area = 0.02  # 0.02

        layer_names = ['L23', 'L4', 'L5', 'L6']
        population_names = ['e', 'i']

        self.full_scale_num_neurons = [
            [int(20683 * area),  # layer 2/3 e
             int(5834 * area)],  # layer 2/3 i
            [int(21915 * area),  # layer 4 e
             int(5479 * area)],  # layer 4 i
            [int(4850 * area),  # layer 5 e
             int(1065 * area)],  # layer 5 i
            [int(14395 * area),  # layer 6 e
             int(2948 * area)]  # layer 6 i
        ]

        # mean EPSP amplitude (mV) for all connections except L4e->L2/3e
        # PSP_e = 0.15
        # mean EPSP amplitude (mv) for L4e->L2/3e connections
        # see p. 801 of the paper, second paragraph under 'Model Parameterization',
        # and the caption to Supplementary Fig. 7
        # PSP_e_23_4 = PSP_e * 2
        # standard deviation of PSC amplitudes relative to mean PSC amplitudes
        # PSC_rel_sd = 0.1
        # IPSP amplitude relative to EPSP amplitude
        # self.g = -4.0

        # whether to use full-scale in-degrees when downscaling the number of neurons
        # When preserve_K is false, the full-scale connection probabilities are used.
        preserve_K = False

        # probabilities for >=1 connection between neurons in the given populations
        # columns correspond to source populations; rows to target populations
        # source      2/3e    2/3i    4e      4i      5e      5i      6e      6i
        conn_probs = [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
                      [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
                      [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
                      [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
                      [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
                      [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
                      [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                      [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443]]

        # mean dendritic delays for excitatory and inhibitory transmission (ms)
        self.delays = [1.5, 0.75]
        # standard deviation relative to mean delays
        self.delay_rel_sd = 0.5
        # connection pattern used in connection calls connecting populations
        self.conn_dict = {'rule': 'fixed_total_number'}
        # weight distribution of connections between populations
        self.weight_dict_exc = {'distribution': 'normal_clipped', 'low': 0.0}
        self.weight_dict_inh = {'distribution': 'normal_clipped', 'high': 0.0}
        # delay distribution of connections between populations
        self.delay_dict = {'distribution': 'normal_clipped', 'low': 0.1}

        # (eta, eps) parameters for each population
        self.gaussian_set_points = [[(-0.008, 0.008),  # 2/3e_e
                                     (-0.03, 0.03)],  # 2/3e_i
                                    [(-0.044, 0.044),  # 4e_e
                                     (-0.059, 0.059)],  # 4e_i
                                    [(-0.075, 0.075),  # 5e_e
                                     (-0.087, 0.087)],  # 5e_i #this might lead to strange connectivity
                                    [(-0.008, 0.008),  # 6e_e
                                     (-0.075, 0.075)]]  # 6e_i

        '''
        In this implementation of structural plasticity, neurons grow
        connection points called synaptic elements. Synapses can be created
        between compatible synaptic elements. The growth of these elements is
        guided by homeostatic rules, defined as growth curves.
        Here we specify the growth curves for synaptic elements of excitatory
        and inhibitory neurons.
        '''
        # Parameters for the synaptic elements
        self.growth_curve_e_e = {
            'growth_curve': "gaussian",
            'growth_rate': self.egr,#0.0018,  # excitatory synaptic elements of Excitatory neurons
            'continuous': False,
        }

        # Parameters for the synaptic elements
        self.growth_curve_e_i = {
            'growth_curve': "gaussian",
            'growth_rate': self.egr, #0.001,  # inhibitory synaptic elements of Excitatory neurons
            'continuous': False,
        }

        # Parameters for the synaptic elements
        self.growth_curve_i_e = {
            'growth_curve': "gaussian",
            'growth_rate': self.igr, #0.0025,  # excitatory synaptic elements of Inhibitory neurons
            'continuous': False,
        }

        # Parameters for the synaptic elements
        self.growth_curve_i_i = {
            'growth_curve': "gaussian",
            'growth_rate': self.igr, #0.001,  # inhibitory synaptic elements of Inhibitory neurons
            'continuous': False,
        }

        '''
        Now we specify the neuron model.
        '''
        self.model_params = {'tau_m': 10.0,  # membrane time constant (ms)
                             'tau_syn_ex': 0.5,  # excitatory synaptic time constant (ms)
                             'tau_syn_in': 0.5,  # inhibitory synaptic time constant (ms)
                             't_ref': 2.0,  # absolute refractory period (ms)
                             'E_L': -65.0,  # resting membrane potential (mV)
                             'V_th': -50.0,  # spike threshold (mV)
                             'C_m': 250.0,  # membrane capacitance (pF)
                             'V_reset': -65.0,  # reset potential (mV)
                             }

        self.nodes_e = [None] * self.regions
        self.nodes_i = [None] * self.regions
        self.loc_e = [[] for i in range(self.regions)]
        self.loc_i = [[] for i in range(self.regions)]
        # Create a list to store mean values
        if nest.Rank() == 0:
            self.mean_fr_e = [[] for i in range(self.regions)]
            self.mean_fr_i = [[] for i in range(self.regions)]
            self.total_connections_e = [[] for i in range(self.regions)]
            self.total_connections_i = [[] for i in range(self.regions)]
            self.last_connections_msg = None
            self.save_state = False

        '''
        We initialize variables for the post-synaptic currents of the
        excitatory, inhibitory and external synapses. These values were
        calculated from a PSP amplitude of 1 for excitatory synapses,
        -1 for inhibitory synapses and 0.11 for external synapses.
        '''
        self.psc_e = 585.0
        self.psc_i = -585.0
        # self.psc_ext = 6.2
        self.psc_ext = 15.0  # 5.85

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

        for x in range(0, self.regions):
            # Excitatory pop, excitatory elems
            gc_e_e = self.growth_curve_e_e.copy()
            gc_e_e['eta'] = self.gaussian_set_points[x][0][0]
            gc_e_e['eps'] = self.gaussian_set_points[x][0][1]
            # Inhibitory pop, inhibitory elems
            gc_i_i = self.growth_curve_i_i.copy()
            gc_i_i['eta'] = self.gaussian_set_points[x][1][0]
            gc_i_i['eps'] = self.gaussian_set_points[x][1][1]
            # Excitatory pop, inhibitory elems
            gc_e_i = self.growth_curve_e_i.copy()
            gc_e_i['eta'] = self.gaussian_set_points[x][0][0]
            gc_e_i['eps'] = self.gaussian_set_points[x][0][1]
            # Inhibitory pop, excitatory elems
            gc_i_e = self.growth_curve_i_e.copy()
            gc_i_e['eta'] = self.gaussian_set_points[x][1][0]
            gc_i_e['eps'] = self.gaussian_set_points[x][1][1]
            synaptic_elements_e = {
                'Den_ex' + str(x): gc_e_e,
                'Den_in' + str(x): gc_e_i,
                'Axon_ex' + str(x): gc_e_e,
            }
            synaptic_elements_i = {
                'Den_ex' + str(x): gc_i_e,
                'Den_in' + str(x): gc_i_i,
                'Axon_in' + str(x): gc_i_i,
            }

            self.nodes_e[x] = nest.Create(self.neuron_model, self.full_scale_num_neurons[x][0], {
                'synaptic_elements': synaptic_elements_e
            })

            self.nodes_i[x] = nest.Create(self.neuron_model, self.full_scale_num_neurons[x][1], {
                'synaptic_elements': synaptic_elements_i
            })
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
            fr_e = nest.GetStatus(self.loc_e[x], 'Ca'),  # Firing rate
            fr_e = self.comm.gather(fr_e, root=0)
            fr_i = nest.GetStatus(self.loc_i[x], 'Ca'),  # Firing rate
            fr_i = self.comm.gather(fr_i, root=0)
            if nest.Rank() == 0:
                mean = numpy.mean([z for x in fr_e for y in x for z in y])
                self.mean_fr_e[x].append(mean*100)
                if mean > 50:
                    self.healthy = 0
                print("Excitatory FR {}".format(mean*100))
                mean = numpy.mean([z for x in fr_i for y in x for z in y])
                self.mean_fr_i[x].append(mean*100)
                if mean > 50:
                    self.healthy = 0
                print("Inhibitory FR {}".format(mean*100))

    def record_connectivity(self):
        for x in range(0, self.regions):
            syn_elems_e = nest.GetStatus(self.loc_e[x], 'synaptic_elements')
            syn_elems_i = nest.GetStatus(self.loc_i[x], 'synaptic_elements')
            sum_neurons_e = sum(neuron['Axon_ex' + str(x)]['z_connected'] for neuron in syn_elems_e)
            sum_neurons_e = self.comm.gather(sum_neurons_e, root=0)
            sum_neurons_i = sum(neuron['Axon_in' + str(x)]['z_connected'] for neuron in syn_elems_i)
            sum_neurons_i = self.comm.gather(sum_neurons_i, root=0)

            if nest.Rank() == 0:
                self.total_connections_i[x].append(sum(sum_neurons_i))
                if self.total_connections_i[x][-1] > 5000000:
                    self.healthy = 0
                self.total_connections_e[x].append(sum(sum_neurons_e))
                if self.total_connections_e[x][-1] > 5000000:
                    self.healthy = 0

    def simulate_(self):

        nest.Simulate(self.record_interval)
        self.record_ca()
        self.record_connectivity()

    def set_update_interval(self):
        nest.SetStructuralPlasticityStatus({'structural_plasticity_update_interval': self.update_interval, })

    def set_growth_rate(self):
        growth_rate_dict = self.growth_rate
        for x in range(0, self.regions):
            if nest.Rank() == 0:
                print("GR" + str(growth_rate_dict[x]))
            synaptic_elements_e = {'growth_rate': growth_rate_dict[x], }
            synaptic_elements_i = {'growth_rate': -growth_rate_dict[x], }
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
        if (nest.Rank() == 0):
            fite = numpy.zeros(self.regions)
            fiti = numpy.zeros(self.regions)
            for x in range(0, self.regions):
                #firing rate of ex neurons is 80% of the fitness
                fite[x] = abs((numpy.mean(self.mean_fr_e[x][-1])/100) - self.gaussian_set_points[x][0][1])/self.gaussian_set_points[x][0][1]

                #firing rate of the inh neurons is 20% of the fitness
                fiti[x] = abs((numpy.mean(self.mean_fr_i[x][-1])/100) - self.gaussian_set_points[x][1][1])/self.gaussian_set_points[x][1][1]
                print("Fitness ex: {:.4f} Fitness in: {:.4f}".format(fite[x],fiti[x]))
            fitness_all = [fite[x]*0.2 + fiti[x]*0.8 for x in range(0,self.regions)]
            fit = 0.0
            for x in range(0,self.regions):
                fit = fit + fitness_all[x]
            if fit != 0.0:
                fitness = [1.0/fit]
        else:
            fitness = None
        fitness = self.comm.bcast(fitness,root=0)
        print(str(nest.Rank()) + " / " + str(fitness[0]))
        return fitness


    def plot_data(self, fitness):
        fig = pl.figure()
        #fig, ax1 = pl.subplots(4,2, sharex=True, sharey=Truei)
        gs = fig.add_gridspec(4, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])
        ax1.axhline(self.gaussian_set_points[0][0][1]*100,
                    linewidth=2.0, color='k')
        ax2.axhline(self.gaussian_set_points[0][1][1]*100,
                    linewidth=2.0, color='k')
        ax3.axhline(self.gaussian_set_points[1][0][1]*100,
                    linewidth=2.0, color='k')
        ax4.axhline(self.gaussian_set_points[1][1][1]*100,
                    linewidth=2.0, color='k')
        ax5.axhline(self.gaussian_set_points[2][0][1]*100,
                    linewidth=2.0, color='k')
        ax6.axhline(self.gaussian_set_points[2][1][1]*100,
                    linewidth=2.0, color='k')
        ax7.axhline(self.gaussian_set_points[3][0][1]*100,
                    linewidth=2.0, color='k')
        ax8.axhline(self.gaussian_set_points[3][1][1]*100,
                    linewidth=2.0, color='k')

        ax1.plot(self.mean_fr_e[0], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_e[0][-1])-self.gaussian_set_points[0][0][1])/self.gaussian_set_points[0][0][1])), linewidth=1.5)
        ax2.plot(self.mean_fr_i[0], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_i[0][-1])-self.gaussian_set_points[0][1][1])/self.gaussian_set_points[0][1][1])), linewidth=1.5)
        ax3.plot(self.mean_fr_e[1], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_e[1][-1])-self.gaussian_set_points[1][0][1])/self.gaussian_set_points[1][0][1])), linewidth=1.5)
        ax4.plot(self.mean_fr_i[1], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_i[1][-1])-self.gaussian_set_points[1][1][1])/self.gaussian_set_points[1][1][1])), linewidth=1.5)
        ax5.plot(self.mean_fr_e[2], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_e[2][-1])-self.gaussian_set_points[2][0][1])/self.gaussian_set_points[2][0][1])), linewidth=1.5)
        ax6.plot(self.mean_fr_i[2], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_i[2][-1])-self.gaussian_set_points[2][1][1])/self.gaussian_set_points[2][1][1])), linewidth=1.5)
        ax7.plot(self.mean_fr_e[3], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_e[3][-1])-self.gaussian_set_points[3][0][1])/self.gaussian_set_points[3][0][1])), linewidth=1.5)
        ax8.plot(self.mean_fr_i[3], color=pl.cm.twilight(0.5+((numpy.mean(self.mean_fr_i[3][-1])-self.gaussian_set_points[3][1][1])/self.gaussian_set_points[3][1][1])), linewidth=1.5)
        #plt.clim(0,1)
        #pl.colorbar()
        #ax1.set_ylim([0, 27.5])
        #ax1.set_xlabel("Time in [steps]")
        #ax1.set_ylabel("Firing rate [Hz]")
        pl.savefig("sp_{:.4f}_{:.4f}_{:.4f}.eps".format(self.egr, self.igr, fitness[0]), format='eps')

        #fig, ax2 = pl.subplots()
        #ax2 = ax1.twinx()
        #ax2.plot(self.total_connections_e[0], color='#9999FFFF',
        #         label='Excitatory connections', linewidth=1.0)
        #ax2.plot(self.total_connections_i[0], color='#0999FFFF',
        #         label='Inhibitory connections', linewidth=1.0,  linestyle='--')

        #ax2.plot(self.total_connections_e[1], color='#FF9999FF',
        #         label='Excitatory connections', linewidth=1.0)
        #ax2.plot(self.total_connections_i[1], color='#0F9999FF',
        #         label='Inhibitory connections', linewidth=1.0, linestyle='--')

        #ax2.plot(self.total_connections_e[2], color='#00CCFFFF',
        #         label='Excitatory connections', linewidth=1.0)
        #ax2.plot(self.total_connections_i[2], color='#F0CCFFFF',
        #         label='Inhibitory connections', linewidth=1.0, linestyle='--')

        #ax2.plot(self.total_connections_e[3], color='#660066FF',
        #         label='Excitatory connections', linewidth=1.0)
        #ax2.plot(self.total_connections_i[3], color='#060066FF',
        #         label='Inhibitory connections', linewidth=1.0, linestyle='--')

        #ax2.set_ylim([0, 2500])
        #ax2.set_ylabel("Connections")
        #ax2.legend(loc=4)
        #ax1.legend(loc=1)
        #pl.savefig("spconns_{:.4f}_{:.4f}_{:.4f}.eps".format(self.egr, self.igr, fitness[0]), format='eps')

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
        nest.EnableStructuralPlasticity()
        start = time.time()
        elapsed = 0
        i_counter = 0
        self.healthy = 1
        while self.healthy and i_counter < 500 and elapsed < 220000:
            self.simulate_()
            elapsed = time.time() - start
            if nest.Rank() == 0:
                print('Iteration ' + str(i_counter) + ' finished. Time: '+ str(elapsed) +'\n')
                sys.stdout.flush()
            self.time.append(i_counter * 1000.0)
            i_counter = i_counter + 1

        fit = self.get_fitness()
        #self.plot_data(fit)
        if nest.Rank() == 0:
            self.plot_data(fit)
            handle_res = open(
                    "/p/scratch/cslns/slns009/L2L/res/fr_" + str(self.id) + "_" + str(
                    trajectory.individual.generation) + ".bin",
                "wb")
            pickle.dump(numpy.array(
                [self.mean_fr_e[0], self.mean_fr_e[1], self.mean_fr_e[2], self.mean_fr_e[3], self.mean_fr_i[0],
                 self.mean_fr_i[1], self.mean_fr_i[2], self.mean_fr_i[3], self.time]), handle_res,
                pickle.HIGHEST_PROTOCOL)
            handle_res.close()
            self.plot_power()
        return(fit)

    def plot_power(self):
        pl.figure()
        for x in range(self.regions):
            freq, POW = self.powerspec(numpy.array([self.mean_fr_e[x], self.time]), 1000.0)
            pl.scatter(POW[1], freq)
            pl.savefig("power_e_{}.eps".format(x), format='eps')
            freq, POW = self.powerspec(numpy.array([self.mean_fr_i[x], self.time]), 1000.0)
            pl.scatter(POW[1], freq)
            pl.savefig("power_i_{}.eps".format(x), format='eps')

    def powerspec(self, data, tbin, Df=None, units=False, N=None):
        '''
        Calculate (smoothed) power spectra of all timeseries in data.
        If units=True, power spectra are averaged across units.
        Note that averaging is done on power spectra rather than data.

        Power spectra are normalized by the length T of the time series -> no scaling with T.
        For a Poisson process this yields:

        **Args**:
           data: numpy.ndarray; 1st axis unit, 2nd axis time
           tbin: float; binsize in ms
           Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
           units: bool; average power spectrum

        **Return**:
           (freq, POW): tuple
           freq: numpy.ndarray; frequencies
           POW: if units=False: 2 dim numpy.ndarray; 1st axis unit, 2nd axis frequency
                if units=True:  1 dim numpy.ndarray; frequency series

        **Examples**:
           >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df)
           Out[1]: (freq,POW)
           >>> POW.shape
           Out[2]: (2,len(analog_sig1))

           >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df, units=True)
           Out[1]: (freq,POW)
           >>> POW.shape
           Out[2]: (len(analog_sig1),)

        '''
        if N is None:
            N = len(data)
        freq, DATA = self.calculate_fft(data, tbin)
        df = freq[1] - freq[0]
        T = tbin * len(freq)
        POW = numpy.power(numpy.abs(DATA), 2)
        assert (len(freq) == len(POW[0]))
        POW *= 1. / T * 1e3  # normalization, power independent of T
        return freq, POW

    def calculate_fft(self, data, tbin):
        '''
        calculate the fouriertransform of data
        [tbin] = ms
        '''
        if len(numpy.shape(data)) > 1:
            n = len(data[0])
            return numpy.fft.fftfreq(n, tbin * 1e-3), numpy.fft.fft(data, axis=1)
        else:
            n = len(data)
            return numpy.fft.fftfreq(n, tbin * 1e-3), numpy.fft.fft(data)


    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        self.bound_gr = [[0, 0], [0, 0]]
        self.bound_gr[0][0] = 0.00005  # -0.001, 0.0001
        self.bound_gr[0][1] = 0.005  # 0.001
        self.bound_gr[1][0] = -0.01
        self.bound_gr[1][1] = -0.0001
        return {'egr': (self.random_state.rand() * (self.bound_gr[0][1] - self.bound_gr[0][0])) + self.bound_gr[0][0],
                'igr': (self.random_state.rand() * (self.bound_gr[0][1] - self.bound_gr[0][0])) + self.bound_gr[0][0],
                }

    def bounding_func(self, individual):
        bound_ind = {'egr': numpy.clip(individual['egr'], a_min=self.bound_gr[0][0], a_max=self.bound_gr[0][1]),
                    'igr': numpy.clip(individual['igr'], a_min=self.bound_gr[0][0], a_max=self.bound_gr[0][1])}
        return bound_ind

