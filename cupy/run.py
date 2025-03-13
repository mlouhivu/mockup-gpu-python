#!/usr/bin/python
# -*- coding: ISO-8859-1 -*-
#---------------------------------------------------------------------------#
# Function: Test GPUs using CuPy + mpi4py.                                  #
# Usage: run.py [options]                                                   #
# Help: python run.py --help                                                #
#---------------------------------------------------------------------------#
import argparse
import logging
import atexit
import platform
import sys
import os

from mpi4py import MPI
import cupy
import numpy

import basic
import memory
#import element
#import blas

class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        print('\nerror: {0}'.format(message))
        sys.exit(1)

def deadlock(comm, rank):
    if (rank == 0):
        a = comm.recv(source=1)
    comm.Barrier()

def print_devices(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    src = rank - 1
    tgt = rank + 1
    if src < 0:
        src = MPI.PROC_NULL
    if tgt >= size:
        tgt = MPI.PROC_NULL
    ok = comm.recv(source=src)
    os.system('echo "MPI rank={0}  host=$(hostname)"; nvidia-smi'.format(rank))
    comm.send(1, dest=tgt)

def run():
    usage = '%(prog)s [options]'
    desc = 'Test GPUs using PyCUDA + mpi4py.'
    parser = ArgumentParser(description=desc, usage=usage)
    parser.add_argument('--deadlock', action='store_true', default=False,
            help='trigger a deliberate MPI deadlock at the end')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display additional information while running')
    parser.add_argument('--debug', action='store_true', default=False,
            help='run in debug mode, i.e. maximum information')

    args = parser.parse_args()

    # set logger format etc.
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s ' + \
            '%(message)s @ %(asctime)s %(module)s line %(lineno)s',
            datefmt='%H:%M:%S')
    # set logging thresholds
    if args.debug:
        logging.getLogger('').setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger('').setLevel(logging.WARNING)
    else:
        logging.getLogger('').setLevel(logging.CRITICAL)
    logging.debug('args: %s' % repr(args))

    # init MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('MPI initialised (rank {0}, size {1})'.format(rank, size))

    # init CUDA
    count = cupy.cuda.runtime.getDeviceCount()
    device = cupy.cuda.Device(rank % count)
    print('MPI rank {0}, node {1}, PCI_BUS_ID {2}, GPU devices {3}'.format(
        rank, platform.node(),
        device.pci_bus_id, count))
    print_devices(comm)

    # run tests
    basic.sum()
    memory.managed()
#    element.axpbyz()
#    element.axpbyz_kernel()
#    element.axpbyz_kernel(complex)
#    blas.axpy()
#    blas.axpy(numpy.float32)

    # hang forever!
    if args.deadlock:
        deadlock(comm, rank)

    # the end.
    return 0

if __name__ == '__main__':
    sys.exit(run())
