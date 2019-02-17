#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 4
#define MAX_MASK_WIDTH 3
#define width 8
#define mask_width 3
#define dimension TILE_SIZE+MAX_MASK_WIDTH-1
__global__ void conv(float*, float*, float*, int, int,int);


