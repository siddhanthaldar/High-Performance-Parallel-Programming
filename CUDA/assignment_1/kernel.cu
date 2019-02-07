__global__ void process_kernel1(float *A, float *B, float *C, int N)
{
	int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int globalThreadId = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;

    if (globalThreadId<N)
    {
        C[globalThreadId] = sin(A[globalThreadId]) + cos(B[globalThreadId]);
    }
}

__global__ void process_kernel2(float *A, float *C, int N)
{
    int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int globalThreadId = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;

    if (globalThreadId<N)
    {
        C[globalThreadId] = log(A[globalThreadId]);
    }
}

__global__ void process_kernel3(float *A, float *C, int N)
{
    int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x+ blockIdx.x;
	int threadNum = threadIdx.z * (blockDim.x* blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
	int globalThreadId = blockNum * (blockDim.x * blockDim.y * blockDim.z) +threadNum;

    if (globalThreadId<N)
    {
        C[globalThreadId] = sqrt(A[globalThreadId]);
    }
}