#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef float EL_TYPE;

__global__ void cuda_matrix_add(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int NUM_ROWS, int NUM_COLS)
{
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < NUM_ROWS && col_index < NUM_COLS)
    {
        size_t index = static_cast<size_t>(row_index) * NUM_COLS + col_index; // A[row_index][col_index]
        OUT[index] = A[index] + B[index];
    }
}

void test_matrix_add(int NUM_ROWS, int NUM_COLS, int ROWS_block_size, int COLS_block_size) 
{
    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the matrices on the host device
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);

    // Initialize the matrices with random values
    for (int i = 0; i < NUM_ROWS; i++)
    {
        for (int j = 0; j < NUM_COLS; j++)
        {
            size_t index = static_cast<size_t>(i) * NUM_COLS + j;
            A[index] = rand() % 100;
            B[index] = rand() % 100;
        }
    }

    // Allocate device memory for a
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS));

    // Transfer the matrices to the device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    // Define the launch grid
    int num_blocks_ROWS = (NUM_ROWS + ROWS_block_size - 1) / ROWS_block_size; // ceil(NUM_ROWS / ROWS_block_size)
    int num_blocks_COLS = (NUM_COLS + COLS_block_size - 1) / COLS_block_size; // ceil(NUM_COLS / COLS_block_size)
    printf("Matrix Add - M: %d, N: %d will be processed by (%d x %d) blocks of size (%d x %d)\n", NUM_ROWS, NUM_COLS, num_blocks_ROWS, num_blocks_COLS, ROWS_block_size, COLS_block_size);
    dim3 grid(num_blocks_COLS, num_blocks_ROWS, 1);
    dim3 block(COLS_block_size, ROWS_block_size, 1);
    // Run the kernel
    cuda_matrix_add<<<grid, block>>>(d_OUT, d_A, d_B, NUM_ROWS, NUM_COLS);

    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Add - Elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to the host
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyDeviceToHost));

    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_OUT));

    // Time the operation 
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    for (int i = 0; i < NUM_ROWS; i++)
    {
        for (int j = 0; j < NUM_COLS; j++)
        {
            size_t index = static_cast<size_t>(i) * NUM_COLS + j;
            if (OUT[index] != A[index] + B[index])
            {
                printf("Error at index (%d, %d): %.2f != %.2f + %.2f\n", i, j, OUT[index], A[index], B[index]);
                exit(1);
            }
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0 + (end_check.tv_usec - start_check.tv_usec) / 1000.0;
    printf("Matrix Add - Check elapsed time: %f ms\n", elapsed);

    printf("Matrix Add - Result OK\n");

    // Free the memory on the host
    free(A);
    free(B);
    free(OUT);
}

int main()
{
    // set your seed
    srand(0);

    test_matrix_add(10000, 10000, 16, 16);

}