#!/usr/bin/env python3

from __future__ import print_function
import sys
import numpy as np
import os.path
import numpy as np
import itertools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pytools
import time
import argparse
import logging

np.set_printoptions(threshold=sys.maxsize)

here = os.path.dirname(os.path.abspath(__file__))

class Parsweep:


    def make_kernel(self, source_file, warp_size, block_dim_x, args, ext_options='', #{{{
            lineinfo=False, nh='nh'):
        with open(source_file, 'r') as fd:
            source = fd.read()
            source = source.replace('M_PI_F', '%ff' % (np.pi, ))
            opts = ['--ptxas-options=-v', ]# '-maxrregcount=32']# '-lineinfo']
            if lineinfo:
                opts.append('-lineinfo')
            opts.append('-DWARP_SIZE=%d' % (warp_size, ))
            opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
            opts.append('-DNH=%s' % (nh, ))
            if ext_options:
                opts.append(ext_options)

            idirs = [here]
            # logger.info('nvcc options %r', opts)
            network_module = SourceModule(
                    source, options=opts, include_dirs=idirs,
                    no_extern_c=True,
                    keep=False,
            )
            mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjffPfS_S_S_S_')
            step_fn = network_module.get_function(mod_func)

        with open('/p/project/cslns/wikicollab/Parsweep_L2L_RateML/covar.c', 'r') as fd:
            source = fd.read()
            opts = ['-ftz=true']  # for faster rsqrtf in corr
            opts.append('-DWARP_SIZE=%d' % (warp_size, ))
            opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
            covar_module = SourceModule(source, options=opts)
            covar_fn = covar_module.get_function('update_cov')
            cov_corr_fn = covar_module.get_function('cov_to_corr')
        return step_fn, covar_fn, cov_corr_fn #}}}

    def cf(self, array):#{{{
        # coerce possibly mixed-stride, double precision array to C-order single precision
        return array.astype(dtype='f', order='C', copy=True)#}}}

    def nbytes(self, data):#{{{
        # count total bytes used in all data arrays
        nbytes = 0
        for name, array in data.items():
            nbytes += array.nbytes
        return nbytes#}}}

    def make_gpu_data(self, data):#{{{
        # put data onto gpu
        gpu_data = {}
        for name, array in data.items():
            gpu_data[name] = gpuarray.to_gpu(self.cf(array))
        return gpu_data#}}}

    def gpu_info(self):
        cmd = "nvidia-smi -q -d MEMORY,UTILIZATION"
        returned_value = os.system(cmd)  # returns the exit code in unix
        print('returned value:', returned_value)

    def run_simulation(self, weights, lengths, params_matrix, speeds, logger, args, n_nodes, n_work_items, n_params, nstep,
                       n_inner_steps,
                       buf_len, states, dt, min_speed):

        # logger.info('caching strategy %r', args.caching)
        if args.test and args.n_time % 200:
            logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing') #}}}

        # setup data#{{{
        data = { 'weights': weights, 'lengths': lengths, 'params': params_matrix.T }
        base_shape = n_work_items,
        for name, shape in dict(
                tavg0=(n_nodes,),
                tavg1=(n_nodes,),
                state=(buf_len, states * n_nodes),
                covar_means=(2 * n_nodes, ),
                covar_cov=(n_nodes, n_nodes, ),
                corr=(n_nodes, n_nodes, ),
                ).items():
            data[name] = np.zeros(shape + base_shape, 'f')

        gpu_data = self.make_gpu_data(data)#{{{
        logger.info('history shape %r', data['state'].shape)
        logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))#}}}

        # setup CUDA stuff
        step_fn, covar_fn, cov_corr_fn = self.make_kernel(
                source_file=args.filename,
                warp_size=32,
                block_dim_x=args.n_coupling,
                args=args,
                ext_options='-DRAND123',
                # caching=args.caching,
                lineinfo=args.lineinfo,
                nh=buf_len,
                # model=args.model,
                )

        # setup simulation
        tic = time.time()
        streams = [drv.Stream() for i in range(32)]
        events = [drv.Event() for i in range(32)]
        tavg_unpinned = []
        tavg = drv.pagelocked_zeros((32, ) + data['tavg0'].shape, dtype=np.float32)

        gridx = args.n_coupling // args.blockszx
        if (gridx == 0):
            gridx = 1;
        gridy = args.n_speed // args.blockszy
        if (gridy == 0):
            gridy = 1;
        final_block_dim = args.blockszx, args.blockszy, 1
        final_grid_dim = gridx, gridy

        # logger.info('final block dim %r', final_block_dim)
        logger.info('final grid dim %r', final_grid_dim)

        # run simulation
        logger.info('submitting work')
        import tqdm
        for i in tqdm.trange(nstep):

            event = events[i % 32]
            stream = streams[i % 32]

            stream.wait_for_event(events[(i - 1) % 32])

            step_fn(np.uintc(i * n_inner_steps), np.uintc(n_nodes), np.uintc(buf_len), np.uintc(n_inner_steps),
                    np.uintc(n_params), np.float32(dt), np.float32(min_speed),
                    gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
                    #gpu_data['tavg%d' % (i%2,)],
                    gpu_data['tavg0'],
                    block=final_block_dim,
                    grid=final_grid_dim,
                    stream=stream)

            event.record(streams[i % 32])

            # TODO check next integrate not zeroing current tavg?
            tavgk = 'tavg%d' % ((i + 1) % 2, )
            if i >= (nstep // 2):
                i_time = i - nstep // 2
                # update_cov (covar_cov is output, tavgk and covar_means are input)
                covar_fn(np.uintc(i_time), np.uintc(n_nodes),
                    gpu_data['covar_cov'], gpu_data['covar_means'], gpu_data[tavgk],
                    block=final_block_dim, grid=final_grid_dim,
                    stream=stream)

            # async wrt. other streams & host, but not this stream.
            if i >= 32:
                stream.synchronize()
                tavg_unpinned.append(tavg[i % 32].copy())

            drv.memcpy_dtoh_async(tavg[i % 32], gpu_data[tavgk].ptr, stream=stream)

            if i == (nstep - 1):
                cov_corr_fn(np.uintc(nstep // 2), np.uintc(n_nodes),
                        gpu_data['covar_cov'], gpu_data['corr'],
                        block = final_block_dim, grid = final_grid_dim,
                        stream = stream)

        # recover uncopied data from pinned buffer
        if nstep > 32:
            for i in range(nstep % 32, 32):
                stream.synchronize()
                tavg_unpinned.append(tavg[i].copy())

        for i in range(nstep % 32):
            stream.synchronize()
            tavg_unpinned.append(tavg[i].copy())

        corr = gpu_data['corr'].get()


        # elapsed = time.time() - tic
        # release pinned memory
        tavg = np.array(tavg_unpinned)
        return tavg, corr