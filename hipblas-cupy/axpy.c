#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>


void daxpy(hipblasHandle_t handle, int n, double a, double *x_, double *y_)
{
    hipblasDaxpy(handle, n, &a, x_, 1, y_, 1);
}

void saxpy(hipblasHandle_t handle, int n, float a, float *x_, float *y_)
{
    hipblasSaxpy(handle, n, &a, x_, 1, y_, 1);
}

PyObject* daxpy_wrapper(PyObject *self, PyObject *args)
{
    int n;
    double a;
    double *x_;
    double *y_;
    hipblasHandle_t handle;

    if (!PyArg_ParseTuple(args, "idnn", &n, &a, &x_, &y_))
        return NULL;

    hipblasCreate(&handle);
    daxpy(handle, n, a, x_, y_);
    hipblasDestroy(handle);

    Py_RETURN_NONE;
}

PyObject* saxpy_wrapper(PyObject *self, PyObject *args)
{
    int n;
    float a;
    float *x_;
    float *y_;
    hipblasHandle_t handle;

    if (!PyArg_ParseTuple(args, "ifnn", &n, &a, &x_, &y_))
        return NULL;

    hipblasCreate(&handle);
    saxpy(handle, n, a, x_, y_);
    hipblasDestroy(handle);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"daxpy",  daxpy_wrapper, METH_VARARGS, "Daxpy"},
    {"saxpy",  saxpy_wrapper, METH_VARARGS, "Saxpy"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "axpy",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__axpy(void)
{
    return PyModule_Create(&module);
}
