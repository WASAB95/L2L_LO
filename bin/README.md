# How to run the use cases

All use cases are marked with `l2l-` and the use case name, e.g. for the reservoir example `l2l-nest-reservoir.py`

The uses cases are as follows:
 1. [`l2l-nest-reservoir.py`](l2l-nest-reservoir.py):  Reservoir computing and digit recognition with NEST
 1. [`l2l-arbor.py`](l2l-arbor.py)
 1. [`l2l-neuroevolution_ant_colony.py`](l2l-neuroevolution_ant_colony.py): Foraging behaviour with Netlogo and NEST or SpikingLab
 1. [](): Fitting functional connectivity with TVB
 1. [`l2l-neuroevolution_mc_nest.py`](l2l-neuroevolution_mc_nest.py): Solving the Mountain Car Task with OpenAI Gym and NEST
 1. [`l2l-nest-sp-simple.py`](l2l-nest-sp-simple.py): Optimizing structural plasticity in NEST, simple case 
 1. [`l2l-nest-sp-micro.py`](l2l-nest-sp-micro.py): Optimizing structural plasticity in NEST, microcircuit
 
 Within the run scripts the results path should be adapted. A default path is given as `../results` which can be changed.
 
 To run on the cluster we put exemplary scripts into the folder `hpc_scripts`.
 The cluster should support `SLURM` directives. 
 There are two files according for every use case, the `run_*` and `set_*` scripts. 
 The `run_*` files invoke the run, while the `set_*` scripts load necessary modules on the HPC. 
 In the `run_*` files, account options have to be set. These may vary from cluster to cluster. We already put a few options which correspond to commands known on most of the super computers in Juelich.
 The `set_*` require the `PYTHONPATH` to e.g. the `L2L`, `JUBE`, `SDICT` folders. They  are indicated by `<path_to_>` where the user has to specify the correct paths.
