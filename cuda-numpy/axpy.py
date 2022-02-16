from _axpy import daxpy
from _mem import Dallocate, deallocate, Dmemcpy_h2d, Dmemcpy_d2h
from math import sin, cos
from numpy import array

n = 10000;
a = 3.4;

# initialise data and calculate reference values on CPU
x = array([sin(i) * 2.3 for i in range(n)], float)
y = array([cos(i) * 1.1 for i in range(n)], float)
y_ref = a * x + y

# allocate + copy initial values
x_ = Dallocate(n)
y_ = Dallocate(n)
Dmemcpy_h2d(x_, x, n);
Dmemcpy_h2d(y_, y, n);

# calculate axpy on GPU
daxpy(n, a, x_, y_);

# copy result back to host and print with reference
print('  initial: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))
Dmemcpy_d2h(y, y_, n);
print('reference: {0} {1} {2} {3} {4} {5}'.format(
    y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[-2], y_ref[-1]))
print('   result: {0} {1} {2} {3} {4} {5}'.format(
    y[0], y[1], y[2], y[3], y[-2], y[-1]))

# deallocate
deallocate(x_)
deallocate(y_)
