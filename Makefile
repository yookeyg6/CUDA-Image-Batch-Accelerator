NVCC ?= nvcc
all:
	$(NVCC) -O2 -std=c++17 -Iinclude src/main.cu src/cuda_processor.cu -o cuda_capstone
clean:
	rm -f cuda_capstone
