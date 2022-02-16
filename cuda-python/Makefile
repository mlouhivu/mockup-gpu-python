CFLAGS := -shared -I$(PYTHONHOME)/include/python3.8 -fPIC
CC := gcc
CUDAFLAGS := --compiler-options '-fPIC -I$(PYTHONHOME)/include/python3.8' --shared -gencode arch=compute_70,code=sm_70
CUDACC := nvcc

cuda := $(patsubst %.cu, _%.so, $(wildcard *.cu))

all: $(cuda) _mem.so

cuda: $(cuda)

_mem.so: mem.c
	$(CC) $(CFLAGS) -lcudart $< -o $@

_%.so: %.cu
	$(CUDACC) $(CUDAFLAGS) $< -o $@

clean:
	rm $(cuda) _mem.so
