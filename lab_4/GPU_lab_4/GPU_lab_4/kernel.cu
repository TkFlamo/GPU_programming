#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// размер блока или размер подматрицы
#define BLOCK_SIZE 32
//тип, который будут иметь элементы матриц
#define BASE_TYPE float 

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__ void matrixMult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Acols, int Bcols)
{
    int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    BASE_TYPE sum = 0;
    for (int k = 0; k < Acols; k++)
        sum += A[i0 + k] * B[k * Bcols + j0];
    int ind = Bcols * (blockDim.y * blockIdx.y +
        threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}


__global__ void matrixMult_shared(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Acols, int Bcols)
{
	// индекс начала первой подматрицы А, которую
	// обрабатывает блок
	int aBegin = Acols * blockDim.y * blockIdx.y;
	// индекс конца подматрицы А, которую обрабатывает блок
	int aEnd = aBegin + Acols - 1;
	// шаг для перебора подматриц А
	int aStep = blockDim.x;
	// индекс начала первой подматрицы В, которую
	// обрабатывает блок
	int bBegin = blockDim.x * blockIdx.x;
	// шаг для перебора подматриц В
	int bStep = blockDim.y * Bcols;
	// Выделение разделяемой памяти для подматриц
	__shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
	// переменная для вычисления элемента подматрицы
	BASE_TYPE sum = 0.0;
	for (int ia = aBegin, ib = bBegin; ia < aEnd; ia +=
		aStep, ib += bStep)
	{
		// загрузка подматриц А и В из глобальной памяти в
		// разделяемую
		as[threadIdx.y][threadIdx.x] = A[ia + Acols *
			threadIdx.y + threadIdx.x];
		bs[threadIdx.y][threadIdx.x] = B[ib + Bcols *
			threadIdx.y + threadIdx.x];
		// синхронизация нитей
		__syncthreads();
		// перемножение двух матриц
		for (int k = 0; k < blockDim.x; k++)
			sum += as[threadIdx.y][k] *
			bs[k][threadIdx.x];
		// синхронизация нитей
		__syncthreads();
	}
	// индекс результирующего элемента в глобальной памяти
	int ind = Bcols * (blockDim.y * blockIdx.y +
		threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	// запись элемента в глобальную память
	C[ind] = sum;
}

// Выравнивание до кратного BLOCK_SIZE
inline int pad(int a) {
    return ((a + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

// Инициализация матрицы с нулями на паддинге
BASE_TYPE* allocateAndInit(int rows, int cols) {
    int paddedRows = pad(rows);
    int paddedCols = pad(cols);
    BASE_TYPE* M = (BASE_TYPE*)calloc(paddedRows * paddedCols, sizeof(BASE_TYPE));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * paddedCols + j] = rand() / (BASE_TYPE)RAND_MAX;
        }
    }
    return M;
}

void matrixMult_shared_exp(int Arows, int Acols, int Bcols) {
    int Brows = Acols;

    BASE_TYPE* h_A = allocateAndInit(Arows, Acols);
    BASE_TYPE* h_B = allocateAndInit(Brows, Bcols);

    int paddedArows = pad(Arows);
    int paddedAcols = pad(Acols);
    int paddedBcols = pad(Bcols);

    std::cout << "Matrix Mutiple Shared Test. Matrix A size: " << paddedAcols << "x" << paddedArows << std::endl;

    //BASE_TYPE* h_C = (BASE_TYPE*)calloc(paddedArows * paddedBcols, sizeof(BASE_TYPE));
    BASE_TYPE* d_A, * d_B, * d_C;

    checkCudaError(cudaMalloc(&d_A, paddedArows * paddedAcols * sizeof(BASE_TYPE)));
    checkCudaError(cudaMalloc(&d_B, paddedAcols * paddedBcols * sizeof(BASE_TYPE)));
    checkCudaError(cudaMalloc(&d_C, paddedArows * paddedBcols * sizeof(BASE_TYPE)));

    checkCudaError(cudaMemcpy(d_A, h_A, paddedArows * paddedAcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, paddedAcols * paddedBcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((Bcols + BLOCK_SIZE - 1) / BLOCK_SIZE, (Arows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    float kernel_time = 0;

    checkCudaError(cudaEventRecord(start));
    matrixMult_shared << <blocks, threads >> > (d_A, d_B, d_C, Acols, Bcols);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Kernel time: " << kernel_time << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    //free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void matrixMult_exp(int Arows, int Acols, int Bcols) {
    int Brows = Acols;

    BASE_TYPE* h_A = allocateAndInit(Arows, Acols);
    BASE_TYPE* h_B = allocateAndInit(Brows, Bcols);

    int paddedArows = pad(Arows);
    int paddedAcols = pad(Acols);
    int paddedBcols = pad(Bcols);

    std::cout << "Matrix Mutiple Test. Matrix A size: " << paddedAcols << "x" << paddedArows << std::endl;

    //BASE_TYPE* h_C = (BASE_TYPE*)calloc(paddedArows * paddedBcols, sizeof(BASE_TYPE));
    BASE_TYPE* d_A, * d_B, * d_C;

    checkCudaError(cudaMalloc(&d_A, paddedArows * paddedAcols * sizeof(BASE_TYPE)));
    checkCudaError(cudaMalloc(&d_B, paddedAcols * paddedBcols * sizeof(BASE_TYPE)));
    checkCudaError(cudaMalloc(&d_C, paddedArows * paddedBcols * sizeof(BASE_TYPE)));

    checkCudaError(cudaMemcpy(d_A, h_A, paddedArows * paddedAcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, paddedAcols * paddedBcols * sizeof(BASE_TYPE), cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((Bcols + BLOCK_SIZE - 1) / BLOCK_SIZE, (Arows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    float kernel_time = 0;

    checkCudaError(cudaEventRecord(start));
    matrixMult << <blocks, threads >> > (d_A, d_B, d_C, Acols, Bcols);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));

    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Kernel time: " << kernel_time << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    //free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    matrixMult_exp(100, 100, 100);
    int tests[3] = { 100, 1000, 10000 };
    for (int i = 0; i < 3; i++) {
        matrixMult_exp(tests[i], tests[i], tests[i]);
        matrixMult_shared_exp(tests[i], tests[i], tests[i]);
    }
    return 0;
}