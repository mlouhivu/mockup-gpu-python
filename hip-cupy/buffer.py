from math import sin, cos

from numpy import array
import cupy

from _buffer import (gpu_daxpy_pointer, gpu_daxpy_buffer, gpu_daxpy_array,
                     cpu_daxpy_pointer, cpu_daxpy_buffer, cpu_daxpy_array)

n = 10000
a = 3.4

# use unified memory
cupy.cuda.set_allocator(cupy.cuda.malloc_managed)

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
y_ref = a * x + y
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[-2], y_ref[-1]))

# allocate + copy initial values
x_ = cupy.asarray(x)
init_y_ = cupy.asarray(y)

# calculate axpy on GPUs and CPUs, access data using direct pointers
y_ = init_y_.copy()
gpu_daxpy_pointer(n, a, x_.data.ptr, y_.data.ptr)
y_gpu_ = y_.copy()
y_ = init_y_.copy()
cpu_daxpy_pointer(n, a, x_.data.ptr, y_.data.ptr)
y_cpu_ = y_.copy()
print('')
print('Pointers')
print('      GPU: {0} {1} {2} {3} {4} {5}'.format(
    y_gpu_[0], y_gpu_[1], y_gpu_[2], y_gpu_[3], y_gpu_[-2], y_gpu_[-1]))
print('      CPU: {0} {1} {2} {3} {4} {5}'.format(
    y_cpu_[0], y_cpu_[1], y_cpu_[2], y_cpu_[3], y_cpu_[-2], y_cpu_[-1]))

# calculate axpy on GPUs and CPUs, access data using Python's Buffer Protocol
try:
    y_ = init_y_.copy()
    gpu_daxpy_buffer(a, x_, y_)
    y_gpu_ = y_.copy()
    y_ = init_y_.copy()
    cpu_daxpy_buffer(a, x_, y_)
    y_cpu_ = y_.copy()
    print('')
    print('Buffers')
    print('      GPU: {0} {1} {2} {3} {4} {5}'.format(
        y_gpu_[0], y_gpu_[1], y_gpu_[2], y_gpu_[3], y_gpu_[-2], y_gpu_[-1]))
    print('      CPU: {0} {1} {2} {3} {4} {5}'.format(
        y_cpu_[0], y_cpu_[1], y_cpu_[2], y_cpu_[3], y_cpu_[-2], y_cpu_[-1]))
except TypeError as err:
    print('')
    print('FAIL: Buffer protocol is not supported.')
    print('  TypeError:', err)

# calculate axpy on GPUs and CPUs, access data using numpy's Array Interface
try:
    _ = x_.__array_interface__
except Exception:
    print('')
    print('FAIL: Array interface is not supported.')
else:
    y_ = init_y_.copy()
    gpu_daxpy_array(a, x_, y_)
    y_gpu_ = y_.copy()
    y_ = init_y_.copy()
    cpu_daxpy_array(a, x_, y_)
    y_cpu_ = y_.copy()
    print('')
    print('Arrays')
    print('      GPU: {0} {1} {2} {3} {4} {5}'.format(
        y_gpu_[0], y_gpu_[1], y_gpu_[2], y_gpu_[3], y_gpu_[-2], y_gpu_[-1]))
    print('      CPU: {0} {1} {2} {3} {4} {5}'.format(
        y_cpu_[0], y_cpu_[1], y_cpu_[2], y_cpu_[3], y_cpu_[-2], y_cpu_[-1]))
