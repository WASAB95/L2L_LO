import nest
import numpy as np
import pandas as pd
def connect_nodes(input_w, input_d, heartbeat_w, heartbeat_d, nodes_w, nodes_d,
                  output_w, output_d):
    syn_spec_dict_nodes = {
        'weight': nodes_w,
        'delay': nodes_d
    }
    syn_spec_dict_heartbeat = {
        'weight': heartbeat_w,
        'delay': heartbeat_d
    }
    syn_spec_dict_input = {
        'weight': input_w,
        'delay': input_d
    }
    syn_spec_dict_output = {
        'weight': output_w,
        'delay': output_d
    }
    conn_dict = {'rule': 'all_to_all',
                 'allow_autapses': False}
    connect_manually(heartbeat, nodes, heartbeat_w, heartbeat_d)
    connect_manually(inpt, nodes, input_w, input_d)
    connect_manually(nodes, outpt, output_w, output_d)
    i = 0
    for source in range(6):
        for target in range(6):
            if source != target:
                w = nodes_w[i]
                d = nodes_d[i]
                syn_spec_nodes = {'weight': w, 'delay': d}
                nest.Connect(nodes[source], nodes[target], 'one_to_one',
                             syn_spec=syn_spec_nodes)
                i += 1    
def connect_manually(sources, targets, weights, delays):
    i = 0
    for source in range(len(sources)):
        for target in range(len(targets)):
            w = weights[i]
            d = delays[i]
            syn_spec = {'weight': w, 'delay': d}
            nest.Connect(sources[source], targets[target], 'one_to_one',
                         syn_spec=syn_spec)   
            i += 1      
def connect_synapses():
    # Connect synapses
    # syn_spec = {'weight': weights, 'delay': delays}
    nest.CopyModel('static_synapse', 'random_synapse')  # syn_spec=syn_spec)
nest.ResetKernel()
nest.set_verbosity('M_ERROR')
nest.SetKernelStatus(
    {
        'resolution': 1.0
    })
#nest.SetKernelStatus({'rng_seed': 123})
# TODO: adapt here parameters
sim_time = 1000000.
dt = 1.
heartbeat_interval = 2.
# heartbeat = nest.Create('poisson_generator', 6)
heartbeat = nest.Create('spike_generator', 1, params={'spike_times': np.linspace(
    1, sim_time, int(sim_time/heartbeat_interval * dt)).round()})
# Activator activates the input neurons, will be manipulated by NetLogo
activator = nest.Create('dc_generator', 5, params={'amplitude': 400000.})
rng = np.random.default_rng(0)
# Create nodes
params = {'t_ref': 1.0}
inpt = nest.Create('iaf_psc_alpha', 5, params=params)
nodes = nest.Create('iaf_psc_alpha', 6, params=params)
outpt = nest.Create('iaf_psc_alpha', 2, params=params)
spike_detector = nest.Create('spike_recorder', 2)
middle_spike_detector = nest.Create('spike_recorder', 6)
input_detector = nest.Create('spike_recorder', 5)
# connect spike generator to input
nest.Connect(activator, inpt, 'one_to_one')
nest.Connect(outpt, spike_detector, 'one_to_one')
nest.Connect(nodes, middle_spike_detector, 'one_to_one')
nest.Connect(inpt, input_detector, 'one_to_one')
# Connect synapses
connect_synapses()
csv = pd.read_csv('individual_config.csv', header=None, na_filter=False)
weights = csv.iloc[0].values * 100.0
delays = csv.iloc[1].values
w_input2nodes = weights[:30] # .reshape(6, 5)
w_heartbeat2nodes = weights[30:36] # .reshape(6, 1)
w_nodes2nodes = weights[36:66]  
w_nodes2output = weights[66:] # .reshape(2, 6)
d_input2nodes = delays[:30] # .reshape(6, 5)
d_heartbeat2nodes = delays[30:36] # .reshape(6, 1)
d_nodes2nodes =  delays[36:66]
d_nodes2output = delays[66:] # .reshape(2, 6).astype(float)
connect_nodes(w_input2nodes, d_input2nodes, w_heartbeat2nodes,
              d_heartbeat2nodes, w_nodes2nodes, d_nodes2nodes, w_nodes2output,
              d_nodes2output)

