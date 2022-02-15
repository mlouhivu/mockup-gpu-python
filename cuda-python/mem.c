#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>

PyObject* Dallocate(PyObject *self, PyObject *args)
{
    int n;
    double *x_;

    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;

    cudaMalloc((void **) &x_, sizeof(double) * n);

    return Py_BuildValue("i", x_);
}

PyObject* deallocate(PyObject *self, PyObject *args)
{
    int n;
    double *x_;

    if (!PyArg_ParseTuple(args, "i", &x_))
        return NULL;

    cudaFree(x_);

    Py_RETURN_NONE;
}

PyObject* Dmemcpy_h2d(PyObject *self, PyObject *args)
{
    int n;
    double *x;
    double *x_;

    if (!PyArg_ParseTuple(args, "nni", &x_, &x, &n))
        return NULL;

    cudaMemcpy(x_, x, sizeof(double) * n, cudaMemcpyHostToDevice);

    Py_RETURN_NONE;
}

PyObject* Dmemcpy_d2h(PyObject *self, PyObject *args)
{
    int n;
    double *x;
    double *x_;

    if (!PyArg_ParseTuple(args, "nni", &x, &x_, &n))
        return NULL;

    cudaMemcpy(x, x_, sizeof(double) * n, cudaMemcpyDeviceToHost);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"Dallocate", Dallocate, METH_VARARGS,
        "cudaMalloc for doubles"},
    {"deallocate", deallocate, METH_VARARGS,
        "cudaFree"},
    {"Dmemcpy_h2d", Dmemcpy_h2d, METH_VARARGS,
        "cudaMemcpy host->device for doubles"},
    {"Dmemcpy_d2h", Dmemcpy_h2d, METH_VARARGS,
        "cudaMemcpy device->host for doubles"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mem",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit__mem(void)
{
    return PyModule_Create(&module);
}
