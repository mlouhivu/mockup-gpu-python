LDFLAGS := -I$(CUDA_INSTALL_ROOT)/include -L$(CUDA_INSTALL_ROOT)/lib64
LIBS := -lcublas -lcudart -lm
CC := gcc

exe := $(patsubst %.c, %, $(wildcard *.c))

all: $(exe)

%: %.c
	$(CC) $(LDFLAGS) $(LIBS) $< -o $@

clean:
	rm $(exe)
