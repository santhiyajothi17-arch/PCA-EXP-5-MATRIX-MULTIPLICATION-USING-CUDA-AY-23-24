# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>AIM:</h3>

To perform Matrix Multiplication using CUDA and check its performance with nvprof.

<h3>ENTER YOUR NAME</h3>
G SANTHIYA
<h3>ENTER YOUR REGISTER NO</h3>
212225230248
<h3>EX. NO</h3>
<h3>DATE</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
TYPE YOUR CODE HERE
```
%%writefile matrix.cu
#include <stdio.h>
#include <cuda.h>

#define SIZE 4
#define BLOCK_SIZE 2

__global__ void matrixMultiply(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < SIZE && col < SIZE)
    {
        int sum = 0;

        for (int k = 0; k < SIZE; k++)
            sum += a[row * SIZE + k] * b[k * SIZE + col];

        c[row * SIZE + col] = sum;
    }
}

int main()
{
    int a[SIZE * SIZE], b[SIZE * SIZE], c[SIZE * SIZE];
    int *da, *db, *dc;

    for (int i = 0; i < SIZE * SIZE; i++)
    {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    cudaMalloc(&da, sizeof(int) * SIZE * SIZE);
    cudaMalloc(&db, sizeof(int) * SIZE * SIZE);
    cudaMalloc(&dc, sizeof(int) * SIZE * SIZE);

    cudaMemcpy(da, a, sizeof(int)*SIZE*SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(int)*SIZE*SIZE, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(2, 2);

    matrixMultiply<<<grid, block>>>(da, db, dc);

    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, sizeof(int)*SIZE*SIZE, cudaMemcpyDeviceToHost);

    printf("Result Matrix C:\n");

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
            printf("%4d", c[i * SIZE + j]);

        printf("\n");
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
```

## OUTPUT:
SHOW YOUR OUTPUT HERE
<img width="407" height="177" alt="image" src="https://github.com/user-attachments/assets/7c0cc36f-941c-4c63-9818-53021039b3a2" />


## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took ______________time and host took ___________time.
