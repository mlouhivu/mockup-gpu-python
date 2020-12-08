CFLAGS := -gencode arch=compute_70,code=sm_70
CC := nvcc

all: axpy axpy-mono dot sum copy copy2d copy3d

axpy: axpy.cu
	$(CC) $(CFLAGS) $< -o $@

axpy-mono: axpy-mono.cu
	$(CC) $(CFLAGS) $< -o $@

dot: dot.cu
	$(CC) $(CFLAGS) $< -o $@

sum: sum.cu
	$(CC) $(CFLAGS) $< -o $@

copy: copy.cu
	$(CC) $(CFLAGS) $< -o $@

copy2d: copy2d.cu
	$(CC) $(CFLAGS) $< -o $@

copy3d: copy3d.cu
	$(CC) $(CFLAGS) $< -o $@
