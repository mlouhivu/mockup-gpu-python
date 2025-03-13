import numba
import numpy

@numba.vectorize(
        ['float32(float32, float32)', 'float64(float64, float64)'],
        target='cuda')
def _sum(a, b):
    return a + b

def managed():
    x = numba.cuda.managed_array(10)
    y = numba.cuda.managed_array(10)
    x.set(numpy.arange(10))
    y.set(numpy.arange(10) * 2)
    z = _sum(x, y)
    print('pure Numba (managed): z = x + y')
    print('  x={0}'.format(x))
    print('  y={0}'.format(y))
    print('  z={0}'.format(z))
    print('')

    y = numpy.arange(10) * 2
    z = _sum(x, y)
    print('Numba (managed) + numpy: z = x + y')
    print('  x={0}'.format(x))
    print('  y={0}'.format(y))
    print('  z={0}'.format(z))
    print('')

if __name__ == '__main__':
    managed()
