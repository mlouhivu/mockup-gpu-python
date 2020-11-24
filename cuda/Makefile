CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

all: axpy axpy-mono dot

axpy: axpy.cu
	$(CC) $(CFLAGS) $< -o $@

axpy-mono: axpy-mono.cu
	$(CC) $(CFLAGS) $< -o $@

dot: dot.cu
	$(CC) $(CFLAGS) $< -o $@
