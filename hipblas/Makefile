INCLUDE := -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hipblas/include
LDFLAGS := $(INCLUDE) -L$(ROCM_PATH)/hipblas/lib -D__HIP_PLATFORM_AMD__
LIBS := -lhipblas
CC := gcc

exe := $(patsubst %.c, %, $(wildcard *.c))

all: $(exe)

%: %.c
	$(CC) $(LDFLAGS) $(LIBS) $< -o $@

clean:
	rm $(exe)
