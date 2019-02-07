#include "headers.h"
/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    // int numElements = 50000;
    int mat_size = 4;
    size_t size = mat_size*mat_size * sizeof(float);
    printf("Doing operations on matrix of size %d x %d\n", mat_size, mat_size);

    float *h_A = (float*)malloc(mat_size*mat_size * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < mat_size; ++i)
    {
        for(int j=0; j<mat_size; j++)
            h_A[i*mat_size+j] = rand()/(float)RAND_MAX;
    }

    for (int i = 0; i < mat_size; ++i)
    {
        for(int j=0; j<mat_size; j++)
            printf("%f ",h_A[i*mat_size+j]);
        printf("\n");
    }
    // Allocate the device input vector A
    // Every function with a "cuda" prefix has a error code returned which can be used to track error
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    ///////////////////////////////////// Operation 1 //////////////////////////////////////////////

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Process Kernel 1 
    dim3 grid1(1,1,1);
    dim3 block1(32,32,1);
    op1<<<grid1,block1>>>(d_A, mat_size);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("\n Swapped Matrix\n");
    for (int i = 0; i < mat_size; ++i)
    {
        for(int j=0; j<mat_size; j++)
            printf("%f ",h_A[i*mat_size+j]);
        printf("\n");
    }

    /////////////////////////////// Operation 2 ////////////////////////////////////////

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Process Kernel 1 
    dim3 grid2(1,1,1);
    dim3 block2(32,32,1);
    op2<<<grid2,block2>>>(d_A, mat_size);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("\n Mirrored Matrix\n");
    for (int i = 0; i < mat_size; ++i)
    {
        for(int j=0; j<mat_size; j++)
            printf("%f ",h_A[i*mat_size+j]);
        printf("\n");
    }


    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

