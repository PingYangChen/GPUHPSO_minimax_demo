CUDA_LIB:=/usr/local/cuda/lib64 -lcuda -lcudart

all: CPU GPU
	g++ testmain.o testgpu.o -o minimax.run -L $(CUDA_LIB)

CPU:
	g++ testmain.c -c -O3

GPU:
	nvcc testgpu.cu -c -arch sm_20 -O3

