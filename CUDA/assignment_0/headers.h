#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void vectorAdd(const float*, const float*, float*, int);

