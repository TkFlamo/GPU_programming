#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int BLOCK_SIZE = 16;

#pragma pack(push, 1)
struct BMPHeader {
    // File header
    uint16_t file_type{ 0x4D42 };          // "BM"
    uint32_t file_size{ 0 };               // Size of file
    uint16_t reserved1{ 0 };
    uint16_t reserved2{ 0 };
    uint32_t offset_data{ 0 };             // Offset to pixel data

    // DIB header
    uint32_t size{ 40 };                   // Header size
    int32_t width{ 0 };                    // Width in pixels
    int32_t height{ 0 };                   // Height in pixels
    uint16_t planes{ 1 };
    uint16_t bit_count{ 24 };              // 24 bits per pixel
    uint32_t compression{ 0 };             // No compression
    uint32_t size_image{ 0 };              // Size of raw bitmap data
    int32_t x_pixels_per_meter{ 0 };
    int32_t y_pixels_per_meter{ 0 };
    uint32_t colors_used{ 0 };
    uint32_t colors_important{ 0 };
};
#pragma pack(pop)

class BMPImage {
private:
    BMPHeader header;
    std::vector<uint8_t> data;

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }

        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        // Verify BMP signature
        if (header.file_type != 0x4D42) {
            std::cerr << "Error: Not a valid BMP file" << std::endl;
            return false;
        }

        if (header.bit_count != 24) {
            std::cerr << "Error: Only 24-bit BMP images are supported" << std::endl;
            return false;
        }

        if (header.compression != 0) {
            std::cerr << "Error: Compressed BMP images are not supported" << std::endl;
            return false;
        }

        uint32_t row_size = ((header.width * 3 + 3) / 4) * 4;
        uint32_t data_size = row_size * header.height;

        data.resize(data_size);
        file.seekg(header.offset_data, std::ios::beg);
        file.read(reinterpret_cast<char*>(data.data()), data_size);

        if (file.fail()) {
            std::cerr << "Error: Failed to read image data" << std::endl;
            return false;
        }

        return true;
    }

    bool save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot create file " << filename << std::endl;
            return false;
        }

        // Update header fields
        uint32_t row_size = ((header.width * 3 + 3) / 4) * 4;
        header.size_image = row_size * header.height;
        header.file_size = sizeof(BMPHeader) + header.size_image;
        header.offset_data = sizeof(BMPHeader);

        file.write(reinterpret_cast<char*>(&header), sizeof(header));
        file.write(reinterpret_cast<char*>(data.data()), data.size());

        if (file.fail()) {
            std::cerr << "Error: Failed to write image data" << std::endl;
            return false;
        }

        return true;
    }

    uint8_t* getData() { return data.data(); }
    const uint8_t* getData() const { return data.data(); }
    size_t getDataSize() const { return data.size(); }
    int getWidth() const { return header.width; }
    int getHeight() const { return header.height; }
    int getChannels() const { return 3; }

    uint32_t getRowSize() const {
        return ((header.width * 3 + 3) / 4) * 4;
    }

    void printInfo() const {
        std::cout << "Image info:" << std::endl;
        std::cout << "  Width: " << header.width << " pixels" << std::endl;
        std::cout << "  Height: " << header.height << " pixels" << std::endl;
        std::cout << "  Bit depth: " << header.bit_count << " bits" << std::endl;
        std::cout << "  Data size: " << data.size() << " bytes" << std::endl;
        std::cout << "  Row size: " << getRowSize() << " bytes" << std::endl;
    }
};

void sobelEdgeDetectionCPU(uint8_t* output, const uint8_t* input,
    int width, int height, int rowSize) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
                // Закрашиваем границу в черный
                if (x < width && y < height) {
                    int idx = y * rowSize + x * 3;
                    output[idx] = 0;
                    output[idx + 1] = 0;
                    output[idx + 2] = 0;
                }
            }
            else {
                // Ядра для оператора Собеля x y
                int sobel_x[3][3] = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
                };

                int sobel_y[3][3] = {
                    {-1, -2, -1},
                    {0, 0, 0},
                    {1, 2, 1}
                };

                float gradient_x = 0.0f;
                float gradient_y = 0.0f;

                // Применяем ядра
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        int idx = py * rowSize + px * 3;

                        uint8_t pixel_value = static_cast <uint8_t>(0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2]);

                        gradient_x += pixel_value * sobel_x[ky + 1][kx + 1];
                        gradient_y += pixel_value * sobel_y[ky + 1][kx + 1];
                    }
                }

                // Вычисляем величину градиента
                float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

                // Преобразуем в цвет
                uint8_t edge_value;
                if (magnitude > 100.0f) {
                    edge_value = 255;
                }
                else if (magnitude > 30.0f) {
                    edge_value = static_cast<uint8_t>(magnitude * 2.0f);
                }
                else {
                    edge_value = 0;
                }

                int out_idx = y * rowSize + x * 3;
                output[out_idx] = edge_value;
                output[out_idx + 1] = edge_value;
                output[out_idx + 2] = edge_value;
            }
        }
    }

}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// Ядро для оператора Собеля
__global__ void sobelEdgeDetectionKernel(uint8_t* output, const uint8_t* input,
    int width, int height, int rowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // Закрашиваем границу в черный
        if (x < width && y < height) {
            int idx = y * rowSize + x * 3;
            output[idx] = 0;
            output[idx + 1] = 0;
            output[idx + 2] = 0;
        }
        return;
    }

    // Ядра для оператора Собеля x y
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    float gradient_x = 0.0f;
    float gradient_y = 0.0f;

    // Применяем ядра
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            int idx = py * rowSize + px * 3;

            uint8_t pixel_value = static_cast <uint8_t>(0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2]);

            gradient_x += pixel_value * sobel_x[ky + 1][kx + 1];
            gradient_y += pixel_value * sobel_y[ky + 1][kx + 1];
        }
    }

    // Вычисляем величину градиента
    float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

    // Преобразуем в цвет
    uint8_t edge_value;
    if (magnitude > 100.0f) {
        edge_value = 255;
    }
    else if (magnitude > 30.0f) {
        edge_value = static_cast<uint8_t>(magnitude * 2.0f);
    }
    else {
        edge_value = 0;
    }

    int out_idx = y * rowSize + x * 3;
    output[out_idx] = edge_value;
    output[out_idx + 1] = edge_value;
    output[out_idx + 2] = edge_value;
}

// Ядро для оператора Собеля
__global__ void sobelEdgeDetectionKernel_shared(uint8_t* output, const uint8_t* input,
    int width, int height, int rowSize) {
    __shared__ uint8_t shared_input[(BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * 3];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tread_x = threadIdx.x;
    int tread_y = threadIdx.y;

    if (x < width && y < height) {
        shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3] = input[y * rowSize + x * 3];
        shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3 + 1] = input[y * rowSize + x * 3 + 1];
        shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3 + 2] = input[y * rowSize + x * 3 + 2];
    }

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        if (tread_x == 0) {
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3] = input[y * rowSize + (x - 1) * 3];
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + 1] = input[y * rowSize + (x - 1) * 3 + 1];
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + 2] = input[y * rowSize + (x - 1) * 3 + 2];
            if (tread_y == 0) {
                shared_input[0] = input[(y - 1) * rowSize + (x - 1) * 3];
                shared_input[1] = input[(y - 1) * rowSize + (x - 1) * 3 + 1];
                shared_input[2] = input[(y - 1) * rowSize + (x - 1) * 3 + 2];
            }
            if (tread_y == (BLOCK_SIZE - 1)) {
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3] = input[(y + 1) * rowSize + (x - 1) * 3];
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + 1] = input[(y + 1) * rowSize + (x - 1) * 3 + 1];
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + 2] = input[(y + 1) * rowSize + (x - 1) * 3 + 2];
            }
        }
        if (tread_x == (BLOCK_SIZE - 1)) {
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3] = input[y * rowSize + (x + 1) * 3];
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3 + 1] = input[y * rowSize + (x + 1) * 3 + 1];
            shared_input[(tread_y + 1) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3 + 2] = input[y * rowSize + (x + 1) * 3 + 2];
            if (tread_y == 0) {
                shared_input[(tread_x + 2) * 3] = input[(y - 1) * rowSize + (x + 1) * 3];
                shared_input[(tread_x + 2) * 3 + 1] = input[(y - 1) * rowSize + (x + 1) * 3 + 1];
                shared_input[(tread_x + 2) * 3 + 2] = input[(y - 1) * rowSize + (x + 1) * 3 + 2];
            }
            if (tread_y == (BLOCK_SIZE - 1)) {
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3] = input[(y + 1) * rowSize + (x + 1) * 3];
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3 + 1] = input[(y + 1) * rowSize + (x + 1) * 3 + 1];
                shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 2) * 3 + 2] = input[(y + 1) * rowSize + (x + 1) * 3 + 2];
            }
        }
        if (tread_y == 0) {
            shared_input[(tread_x + 1) * 3] = input[(y - 1) * rowSize + x * 3];
            shared_input[(tread_x + 1) * 3 + 1] = input[(y - 1) * rowSize + x * 3 + 1];
            shared_input[(tread_x + 1) * 3 + 2] = input[(y - 1) * rowSize + x * 3 + 2];
        }
        if (tread_y == (BLOCK_SIZE - 1)) {
            shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3] = input[(y + 1) * rowSize + x * 3];
            shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3 + 1] = input[(y + 1) * rowSize + x * 3 + 1];
            shared_input[(tread_y + 2) * (BLOCK_SIZE + 2) * 3 + (tread_x + 1) * 3 + 2] = input[(y + 1) * rowSize + x * 3 + 2];
        }

    }

    __syncthreads();

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // Закрашиваем границу в черный
        if (x < width && y < height) {
            int idx = y * rowSize + x * 3;
            output[idx] = 0;
            output[idx + 1] = 0;
            output[idx + 2] = 0;
        }
        return;
    }

    // Ядра для оператора Собеля x y
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    float gradient_x = 0.0f;
    float gradient_y = 0.0f;

    // Применяем ядра
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = tread_x + kx;
            int py = tread_y + ky;
            int idx = (py + 1) * (BLOCK_SIZE + 2) * 3 + (px + 1) * 3;

            uint8_t pixel_value = static_cast <uint8_t>(0.299f * shared_input[idx] + 0.587f * shared_input[idx + 1] + 0.114f * shared_input[idx + 2]);

            gradient_x += pixel_value * sobel_x[ky + 1][kx + 1];
            gradient_y += pixel_value * sobel_y[ky + 1][kx + 1];
        }
    }

    // Вычисляем величину градиента
    float magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y);

    // Преобразуем в цвет
    uint8_t edge_value;
    if (magnitude > 100.0f) {
        edge_value = 255;
    }
    else if (magnitude > 30.0f) {
        edge_value = static_cast<uint8_t>(magnitude * 2.0f);
    }
    else {
        edge_value = 0;
    }

    int out_idx = y * rowSize + x * 3;
    output[out_idx] = edge_value;
    output[out_idx + 1] = edge_value;
    output[out_idx + 2] = edge_value;
}

void applyEdgeDetectionCPU(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = image.getWidth();
    int height = image.getHeight();
    int rowSize = image.getRowSize();
    size_t data_size = image.getDataSize();

    std::cout << "Starting CUDA edge detection..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Data size: " << data_size << " bytes" << std::endl;

    uint8_t* output = (uint8_t*) malloc(data_size);

    float kernel_time = 0;

    std::cout << "Applying Sobel edge detection..." << std::endl;
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    sobelEdgeDetectionCPU(output, image.getData(), width, height, rowSize);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - cpu_start_time);
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    memcpy(image.getData(), output, data_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
}

void applyEdgeDetection(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = image.getWidth();
    int height = image.getHeight();
    int rowSize = image.getRowSize();
    size_t data_size = image.getDataSize();

    std::cout << "Starting CUDA edge detection..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Data size: " << data_size << " bytes" << std::endl;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    uint8_t* d_input, * d_output;
    checkCudaError(cudaMalloc(&d_input, data_size));
    checkCudaError(cudaMalloc(&d_output, data_size));

    checkCudaError(cudaMemcpy(d_input, image.getData(), data_size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    float kernel_time = 0;

    std::cout << "Applying Sobel edge detection..." << std::endl;
    checkCudaError(cudaEventRecord(start));
    sobelEdgeDetectionKernel << <gridDim, blockDim >> > (d_output, d_input, width, height, rowSize);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Kernel time: " << kernel_time << std::endl;
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<uint8_t> output_data(data_size);
    checkCudaError(cudaMemcpy(output_data.data(), d_output, data_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));

    memcpy(image.getData(), output_data.data(), data_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
}

void applyEdgeDetection_shared(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = image.getWidth();
    int height = image.getHeight();
    int rowSize = image.getRowSize();
    size_t data_size = image.getDataSize();

    std::cout << "Starting CUDA edge detection..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Data size: " << data_size << " bytes" << std::endl;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    uint8_t* d_input, * d_output;
    checkCudaError(cudaMalloc(&d_input, data_size));
    checkCudaError(cudaMalloc(&d_output, data_size));

    checkCudaError(cudaMemcpy(d_input, image.getData(), data_size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    float kernel_time = 0;

    std::cout << "Applying shared Sobel edge detection..." << std::endl;
    checkCudaError(cudaEventRecord(start));
    sobelEdgeDetectionKernel_shared << <gridDim, blockDim, (BLOCK_SIZE + 2)* (BLOCK_SIZE + 2) * 3 >> > (d_output, d_input, width, height, rowSize);
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&kernel_time, start, stop));
    std::cout << "Kernel time: " << kernel_time << std::endl;
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<uint8_t> output_data(data_size);
    checkCudaError(cudaMemcpy(output_data.data(), d_output, data_size, cudaMemcpyDeviceToHost));

    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));

    memcpy(image.getData(), output_data.data(), data_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
}

int main() {

    std::string input_file = "test_3.bmp";
    std::string output_file = "output_3.bmp";
    std::string output_file_CPU = "output_3_CPU.bmp";
    std::string output_file_shared = "output_3_shared.bmp";

    BMPImage image;

    std::cout << "Loading image: " << input_file << std::endl;
    if (!image.load(input_file)) {
        std::cerr << "Error: Failed to load image" << std::endl;
        return 1;
    }

    image.printInfo();

    std::cout << "\nApplying shared edge detection on CPU..." << std::endl;
    applyEdgeDetectionCPU(image);

    std::cout << "\nSaving result to: " << output_file_CPU << std::endl;
    if (!image.save(output_file_CPU)) {
        std::cerr << "Error: Failed to save image" << std::endl;
        return 1;
    }

    std::cout << "Result saved successfully" << std::endl;

    if (!image.load(input_file)) {
        std::cerr << "Error: Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "\nApplying edge detection on GPU..." << std::endl;
    applyEdgeDetection(image);

    std::cout << "\nSaving result to: " << output_file << std::endl;
    if (!image.save(output_file)) {
        std::cerr << "Error: Failed to save image" << std::endl;
        return 1;
    }

    std::cout << "Result saved successfully" << std::endl;

    if (!image.load(input_file)) {
        std::cerr << "Error: Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "\nApplying shared edge detection on GPU..." << std::endl;
    applyEdgeDetection_shared(image);

    std::cout << "\nSaving result to: " << output_file_shared << std::endl;
    if (!image.save(output_file_shared)) {
        std::cerr << "Error: Failed to save image" << std::endl;
        return 1;
    }

    std::cout << "Result saved successfully" << std::endl;

    return 0;
}