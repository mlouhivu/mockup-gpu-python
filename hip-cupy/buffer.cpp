#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void daxpy_(int n, double a, double *x, double *y)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        y[tid] += a * x[tid];
    }
}

void daxpy(int n, double a, double *x_, double *y_)
{
    hipLaunchKernelGGL(daxpy_, 32, 256, 0, 0, n, a, x_, y_);
}

void cpu_daxpy(int n, double a, double *x, double *y)
{
    int i;
    for (i=0; i < n; i++) {
        y[i] += a * x[i];
    }
}

extern "C"
PyObject* gpu_daxpy_pointer(PyObject *self, PyObject *args)
{
    int n;
    double a;
    double *x_;
    double *y_;

    if (!PyArg_ParseTuple(args, "idnn", &n, &a, &x_, &y_))
        return NULL;

    daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* cpu_daxpy_pointer(PyObject *self, PyObject *args)
{
    int n;
    double a;
    double *x_;
    double *y_;

    if (!PyArg_ParseTuple(args, "idnn", &n, &a, &x_, &y_))
        return NULL;

    cpu_daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* gpu_daxpy_buffer(PyObject *self, PyObject *args)
{
    double a;
    PyObject* x;
    PyObject* y;

    if (!PyArg_ParseTuple(args, "dOO", &a, &x, &y))
        return NULL;

    /* access data using Python's C API Buffer Protocol:
       https://docs.python.org/dev/c-api/buffer.html
       https://peps.python.org/pep-3118/
     */
    Py_buffer buffer_x;
    Py_buffer buffer_y;
    if (PyObject_GetBuffer(
                x, &buffer_x, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    if (PyObject_GetBuffer(
                y, &buffer_y, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    int n = buffer_x.shape[0];
    double *x_ = (double *) buffer_x.buf;
    double *y_ = (double *) buffer_y.buf;

    daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* cpu_daxpy_buffer(PyObject *self, PyObject *args)
{
    double a;
    PyObject* x;
    PyObject* y;

    if (!PyArg_ParseTuple(args, "dOO", &a, &x, &y))
        return NULL;

    /* access data using Python's C API Buffer Protocol:
       https://docs.python.org/dev/c-api/buffer.html
       https://peps.python.org/pep-3118/
     */
    Py_buffer buffer_x;
    Py_buffer buffer_y;
    if (PyObject_GetBuffer(
                x, &buffer_x, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    if (PyObject_GetBuffer(
                y, &buffer_y, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    int n = buffer_x.shape[0];
    double *x_ = (double *) buffer_x.buf;
    double *y_ = (double *) buffer_y.buf;

    cpu_daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* gpu_daxpy_array(PyObject *self, PyObject *args)
{
    double a;
    PyArrayObject* x;
    PyArrayObject* y;

    if (!PyArg_ParseTuple(args, "dOO", &a, &x, &y))
        return NULL;

    /* access data using numpy's Array Interface API:
       https://numpy.org/doc/stable/reference/arrays.interface.html */
    int n = PyArray_DIMS(x)[0];
    double *x_ = (double *) PyArray_DATA(x);
    double *y_ = (double *) PyArray_DATA(y);

    daxpy(n, a, x_, y_);

    Py_RETURN_NONE;
}

extern "C"
PyObject* cpu_daxpy_array(PyObject *self, PyObject *args)
{
    double a;
    PyArrayObject* x;
    PyArrayObject* y;

    if (!PyArg_ParseTuple(args, "dOO", &a, &x, &y))
        return NULL;

    /* access data using numpy's Array Interface API:
       https://numpy.org/doc/stable/reference/arrays.interface.html */
    int n = PyArray_DIMS(x)[0];
    double *data_x = (double *) PyArray_DATA(x);
    double *data_y = (double *) PyArray_DATA(y);

    cpu_daxpy(n, a, data_x, data_y);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"gpu_daxpy_pointer", gpu_daxpy_pointer, METH_VARARGS, "GPU Daxpy, pointers"},
    {"cpu_daxpy_pointer", cpu_daxpy_pointer, METH_VARARGS, "CPU Daxpy, pointers"},
    {"gpu_daxpy_buffer", gpu_daxpy_buffer, METH_VARARGS, "GPU Daxpy, buffers"},
    {"cpu_daxpy_buffer", cpu_daxpy_buffer, METH_VARARGS, "CPU Daxpy, buffers"},
    {"gpu_daxpy_array", gpu_daxpy_array, METH_VARARGS, "GPU Daxpy, numpy arrays"},
    {"cpu_daxpy_array", cpu_daxpy_array, METH_VARARGS, "CPU Daxpy, numpy arrays"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "buffer",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__buffer(void)
{
    return PyModule_Create(&module);
}
