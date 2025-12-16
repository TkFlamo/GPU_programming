#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

constexpr int N = 100000000;
constexpr int BLOCK_SIZE = 1024;
constexpr float PI = 3.1415927f;
constexpr double PI_DOUBLE = 3.141592653589793;

__global__ void init_array_sin(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = sin((idx % 360) * PI / 180.0f);
    }
}

// Ядро с использованием sinf (single precision)
__global__ void init_array_sinf(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = sinf((idx % 360) * PI / 180.0f);
    }
}

// Ядро с использованием __sin (intrinsic, приближенное вычисление)
__global__ void init_array_sin_intrinsic(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = __sinf((idx % 360) * PI / 180.0f);
    }
}

// Ядро с использованием sin (double precision)
__global__ void init_array_sin_double(double* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = sin((idx % 360) * PI_DOUBLE / 180.0);
    }
}

// Ядро с использованием sinf (double precision)
__global__ void init_array_sinf_double(double* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = sinf((idx % 360) * PI_DOUBLE / 180.0);
    }
}

// Ядро с использованием __sin (double precision)
__global__ void init_array_sin_intrinsic_double(double* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = __sinf((idx % 360) * PI_DOUBLE / 180.0);
    }
}

// Функция для проверки ошибок CUDA
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Вычисление ошибки на CPU
template<typename T>
double calculate_error(T* arr, int n) {
    double total_error = 0.0;
    for (int i = 0; i < n; i++) {
        double expected = sin((i % 360) * PI_DOUBLE / 180.0);
        total_error += abs(expected - arr[i]);
    }
    return total_error / n;
}

int main() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Device name : %s\n", deviceProp.name);
    printf("Total global memory : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %d\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n", deviceProp.regsPerBlock);
    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %d\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z =% d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Total constant memory: %d\n", deviceProp.totalConstMem);
    std::cout << "Array shape: " << N << std::endl;
    std::cout << "==============================================" << std::endl;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // Выделение памяти на CPU
    float* host_float = new float[N];
    double* host_double = new double[N];

    // Выделение памяти на GPU
    float* dev_float;
    double* dev_double;

    checkCudaError(cudaMalloc(&dev_float, N * sizeof(float)));
    checkCudaError(cudaMalloc(&dev_double, N * sizeof(double)));

    // Расчет количества блоков
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float kernel_time = 0;

    // Тест 1: float с sin
    std::cout << "\n1. float sin:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sin << <grid, block >> > (dev_float, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_float, dev_float, N * sizeof(float), cudaMemcpyDeviceToHost));
    double error1 = calculate_error(host_float, N);
    std::cout << "Mean error: " << error1 << std::endl;

    // Тест 2: float с sinf
    std::cout << "\n2. float sinf:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sinf << <grid, block >> > (dev_float, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_float, dev_float, N * sizeof(float), cudaMemcpyDeviceToHost));
    double error2 = calculate_error(host_float, N);
    std::cout << "Mean error: " << error2 << std::endl;

    // Тест 3: float с __sin
    std::cout << "\n3. float __sinf:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sin_intrinsic << <grid, block >> > (dev_float, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_float, dev_float, N * sizeof(float), cudaMemcpyDeviceToHost));
    double error3 = calculate_error(host_float, N);
    std::cout << "Mean error: " << error3 << std::endl;

    // Тест 4: double с sin
    std::cout << "\n4. double sin:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sin_double << <grid, block >> > (dev_double, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_double, dev_double, N * sizeof(double), cudaMemcpyDeviceToHost));
    double error4 = calculate_error(host_double, N);
    std::cout << "Mean error: " << error4 << std::endl;

    // Тест 5: double с sinf
    std::cout << "\n4. double sinf:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sinf_double << <grid, block >> > (dev_double, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_double, dev_double, N * sizeof(double), cudaMemcpyDeviceToHost));
    double error5 = calculate_error(host_double, N);
    std::cout << "Mean error: " << error5 << std::endl;

    // Тест 6: double с __sinf
    std::cout << "\n4. double __sinf:" << std::endl;
    checkCudaError(cudaEventRecord(start));
    init_array_sin_intrinsic_double << <grid, block >> > (dev_double, N);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Time: " << kernel_time << std::endl;
    checkCudaError(cudaMemcpy(host_double, dev_double, N * sizeof(double), cudaMemcpyDeviceToHost));
    double error6 = calculate_error(host_double, N);
    std::cout << "Mean error: " << error6 << std::endl;

    // Освобождение памяти
    delete[] host_float;
    delete[] host_double;
    cudaFree(dev_float);
    cudaFree(dev_double);

    return 0;
}