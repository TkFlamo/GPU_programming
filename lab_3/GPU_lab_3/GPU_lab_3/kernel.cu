#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int BLOCK_SIZE = 32;
constexpr int KERNEL_SIZE = 3;

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

    // Создание подизображения
    BMPImage createSubImage(int startY, int endY) const {
        BMPImage subImage;
        subImage.header = header;
        subImage.header.height = endY - startY;

        uint32_t row_size = getRowSize();
        uint32_t sub_data_size = row_size * subImage.header.height;
        subImage.header.size_image = sub_data_size;
        subImage.header.file_size = sizeof(BMPHeader) + sub_data_size;

        subImage.data.resize(sub_data_size);

        for (int y = 0; y < subImage.header.height; y++) {
            int src_y = startY + y;
            if (src_y >= 0 && src_y < header.height) {
                const uint8_t* src_row = &data[src_y * row_size];
                uint8_t* dst_row = &subImage.data[y * row_size];
                memcpy(dst_row, src_row, row_size);
            }
        }

        return subImage;
    }

    // Вставка подизображения в текущее изображение
    void insertSubImage(const BMPImage& subImage, int startY) {
        uint32_t row_size = getRowSize();
        for (int y = 0; y < subImage.header.height; y++) {
            int dst_y = startY + y;
            if (dst_y >= 0 && dst_y < header.height) {
                const uint8_t* src_row = &subImage.data[y * row_size];
                uint8_t* dst_row = &data[dst_y * row_size];
                memcpy(dst_row, src_row, row_size);
            }
        }
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

void applyEdgeDetectionCPU(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = image.getWidth();
    int height = image.getHeight();
    int rowSize = image.getRowSize();
    size_t data_size = image.getDataSize();

    std::cout << "Starting CUDA edge detection..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Data size: " << data_size << " bytes" << std::endl;

    uint8_t* output = (uint8_t*)malloc(data_size);

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

// Функция для обработки части изображения на GPU
void processImagePartOnGPU(BMPImage& imagePart, cudaStream_t stream) {
    int width = imagePart.getWidth();
    int height = imagePart.getHeight();
    int rowSize = imagePart.getRowSize();
    size_t data_size = imagePart.getDataSize();

    // Выделяем память на устройстве
    uint8_t* d_input, * d_output;
    checkCudaError(cudaMalloc(&d_input, data_size));
    checkCudaError(cudaMalloc(&d_output, data_size));

    // Копируем данные на устройство
    checkCudaError(cudaMemcpyAsync(d_input, imagePart.getData(), data_size,
        cudaMemcpyHostToDevice, stream));

    // Настраиваем размеры блоков и гридов
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    // Запускаем ядро
    sobelEdgeDetectionKernel << <gridDim, blockDim, 0, stream >> > (
        d_output, d_input, width, height, rowSize);

    // Проверяем ошибки
    checkCudaError(cudaGetLastError());

    // Копируем результат обратно
    std::vector<uint8_t> output_data(data_size);
    checkCudaError(cudaMemcpyAsync(output_data.data(), d_output, data_size,
        cudaMemcpyDeviceToHost, stream));

    // Синхронизируем stream
    checkCudaError(cudaStreamSynchronize(stream));

    // Копируем результат в изображение
    memcpy(imagePart.getData(), output_data.data(), data_size);

    // Освобождаем память устройства
    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));
}

void applyEdgeDetectionTwoStreamsSeparateImages(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int overlap = KERNEL_SIZE / 2;

    int width = image.getWidth();
    int height = image.getHeight();

    std::cout << "Starting CUDA edge detection with 2 streams and separate images..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Using overlap of " << overlap << " rows for correct border processing" << std::endl;

    // Разделяем изображение на 2 части с перекрытием
    int midY = height / 2;

    // Первая часть: верхняя с перекрытием
    int startY1 = 0;
    int endY1 = midY + overlap;
    if (endY1 > height) endY1 = height;

    // Вторая часть: нижняя с перекрытием
    int startY2 = midY - overlap;
    if (startY2 < 0) startY2 = 0;
    int endY2 = height;

    std::cout << "Creating separate images for processing..." << std::endl;
    std::cout << "Part 1: rows " << startY1 << " to " << endY1 << " (height: " << (endY1 - startY1) << ")" << std::endl;
    std::cout << "Part 2: rows " << startY2 << " to " << endY2 << " (height: " << (endY2 - startY2) << ")" << std::endl;

    // Создаем два отдельных изображения
    BMPImage imagePart1 = image.createSubImage(startY1, endY1);
    BMPImage imagePart2 = image.createSubImage(startY2, endY2);

    // Создаем 2 streams
    cudaStream_t stream1, stream2;
    checkCudaError(cudaStreamCreate(&stream1));
    checkCudaError(cudaStreamCreate(&stream2));

    // Создаем события для замера времени
    cudaEvent_t start1, stop1, start2, stop2;
    checkCudaError(cudaEventCreate(&start1));
    checkCudaError(cudaEventCreate(&stop1));
    checkCudaError(cudaEventCreate(&start2));
    checkCudaError(cudaEventCreate(&stop2));

    std::cout << "Processing images in parallel streams..." << std::endl;

    auto start_p_time = std::chrono::high_resolution_clock::now();

    // Запускаем обработку в двух потоках параллельно
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            auto start_p1_time = std::chrono::high_resolution_clock::now();
            checkCudaError(cudaEventRecord(start1, stream1));
            processImagePartOnGPU(imagePart1, stream1);
            checkCudaError(cudaEventRecord(stop1, stream1));
            checkCudaError(cudaStreamSynchronize(stream1));
            auto end_p1_time = std::chrono::high_resolution_clock::now();
            auto duration_p1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_p1_time - start_p1_time);

            std::cout << "Parallel_1 processing time: " << duration_p1.count() << " ms" << std::endl;
        }

#pragma omp section
        {
            checkCudaError(cudaEventRecord(start2, stream2));
            processImagePartOnGPU(imagePart2, stream2);
            checkCudaError(cudaEventRecord(stop2, stream2));
            checkCudaError(cudaStreamSynchronize(stream2));
        }
    }

    checkCudaError(cudaStreamSynchronize(stream1));
    checkCudaError(cudaStreamSynchronize(stream2));

    auto end_p_time = std::chrono::high_resolution_clock::now();
    auto duration_p = std::chrono::duration_cast<std::chrono::milliseconds>(end_p_time - start_p_time);

    std::cout << "Parallel processing time: " << duration_p.count() << " ms" << std::endl;

    // Получаем время выполнения
    float kernel_time1 = 0, kernel_time2 = 0;
    checkCudaError(cudaEventElapsedTime(&kernel_time1, start1, stop1));
    checkCudaError(cudaEventElapsedTime(&kernel_time2, start2, stop2));

    std::cout << "Stream 1 processing time: " << kernel_time1 << " ms" << std::endl;
    std::cout << "Stream 2 processing time: " << kernel_time2 << " ms" << std::endl;

    // Собираем результат обратно
    std::cout << "Merging processed parts..." << std::endl;

    // Вставляем обработанные части, исключая перекрытие
    image.insertSubImage(imagePart1, 0); // Вставляем всю верхнюю часть

    // Для нижней части исключаем перекрытие
    BMPImage imagePart2NoOverlap = imagePart2.createSubImage(overlap, imagePart2.getHeight());
    image.insertSubImage(imagePart2NoOverlap, midY);

    // Освобождаем ресурсы
    checkCudaError(cudaStreamDestroy(stream1));
    checkCudaError(cudaStreamDestroy(stream2));
    checkCudaError(cudaEventDestroy(start1));
    checkCudaError(cudaEventDestroy(stop1));
    checkCudaError(cudaEventDestroy(start2));
    checkCudaError(cudaEventDestroy(stop2));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Total processing time with separate images: " << duration.count() << " ms" << std::endl;
}

// Структура для передачи данных в поток
struct ThreadData {
    BMPImage* imagePart;
    std::vector<uint8_t>* outputData;
    float kernelTime;
    std::thread::id threadId;
    std::mutex* coutMutex;
};

// Функция, выполняемая в каждом потоке
void processImagePartThread(ThreadData* threadData) {
    threadData->threadId = std::this_thread::get_id();

    auto start_time = std::chrono::high_resolution_clock::now();

    BMPImage& imagePart = *(threadData->imagePart);
    int width = imagePart.getWidth();
    int height = imagePart.getHeight();
    int rowSize = imagePart.getRowSize();
    size_t data_size = imagePart.getDataSize();

    // Создаем свой CUDA stream для этого потока
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));

    // Выделяем память на GPU
    uint8_t* d_input, * d_output;
    checkCudaError(cudaMalloc(&d_input, data_size));
    checkCudaError(cudaMalloc(&d_output, data_size));

    // Копируем данные на GPU
    checkCudaError(cudaMemcpyAsync(d_input, imagePart.getData(), data_size,
        cudaMemcpyHostToDevice, stream));

    // Настраиваем размеры блоков и гридов
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    // Создаем события для измерения времени ядра
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));

    // Запускаем ядро
    checkCudaError(cudaEventRecord(start, stream));
    sobelEdgeDetectionKernel << <gridDim, blockDim, 0, stream >> > (
        d_output, d_input, width, height, rowSize);
    checkCudaError(cudaEventRecord(stop, stream));

    // Синхронизируем stream
    checkCudaError(cudaStreamSynchronize(stream));

    // Получаем время выполнения ядра
    checkCudaError(cudaEventElapsedTime(&threadData->kernelTime, start, stop));

    // Копируем результат обратно
    threadData->outputData->resize(data_size);
    checkCudaError(cudaMemcpyAsync(threadData->outputData->data(), d_output, data_size,
        cudaMemcpyDeviceToHost, stream));

    // Синхронизируем stream
    checkCudaError(cudaStreamSynchronize(stream));

    // Освобождаем ресурсы
    checkCudaError(cudaFree(d_input));
    checkCudaError(cudaFree(d_output));
    checkCudaError(cudaStreamDestroy(stream));
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Выводим информацию о потоке
    {
        std::lock_guard<std::mutex> lock(*(threadData->coutMutex));
        std::cout << "Thread " << threadData->threadId << ": " << std::endl;
        std::cout << "  Part size: " << width << "x" << height << std::endl;
        std::cout << "  Kernel time: " << threadData->kernelTime << " ms" << std::endl;
        std::cout << "  Total thread time: " << duration.count() << " ms" << std::endl;
    }
}

void applyEdgeDetectionTwoThreads(BMPImage& image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int width = image.getWidth();
    int height = image.getHeight();
    int overlap = KERNEL_SIZE / 2;

    std::cout << "Starting CUDA edge detection with 2 threads..." << std::endl;
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Using overlap of " << overlap << " rows for correct border processing" << std::endl;

    // Разделяем изображение на 2 части с перекрытием
    int midY = height / 2;

    // Первая часть: верхняя с перекрытием
    int startY1 = 0;
    int endY1 = midY + overlap;
    if (endY1 > height) endY1 = height;

    // Вторая часть: нижняя с перекрытием
    int startY2 = midY - overlap;
    if (startY2 < 0) startY2 = 0;
    int endY2 = height;

    std::cout << "Creating separate images for processing..." << std::endl;
    std::cout << "Part 1: rows " << startY1 << " to " << endY1
        << " (height: " << (endY1 - startY1) << ")" << std::endl;
    std::cout << "Part 2: rows " << startY2 << " to " << endY2
        << " (height: " << (endY2 - startY2) << ")" << std::endl;

    // Создаем два отдельных изображения
    BMPImage imagePart1 = image.createSubImage(startY1, endY1);
    BMPImage imagePart2 = image.createSubImage(startY2, endY2);

    // Подготавливаем данные для потоков
    ThreadData threadData1, threadData2;
    std::vector<uint8_t> outputData1, outputData2;
    std::mutex coutMutex;

    threadData1.imagePart = &imagePart1;
    threadData1.outputData = &outputData1;
    threadData1.coutMutex = &coutMutex;

    threadData2.imagePart = &imagePart2;
    threadData2.outputData = &outputData2;
    threadData2.coutMutex = &coutMutex;

    std::cout << "Launching threads..." << std::endl;

    auto start_p_time = std::chrono::high_resolution_clock::now();

    // Запускаем потоки
    std::thread thread1(processImagePartThread, &threadData1);
    std::thread thread2(processImagePartThread, &threadData2);

    // Ждем завершения потоков
    thread1.join();
    thread2.join();

    auto end_p_time = std::chrono::high_resolution_clock::now();
    auto duration_p = std::chrono::duration_cast<std::chrono::milliseconds>(end_p_time - start_p_time);

    std::cout << "Parallel processing time: " << duration_p.count() << " ms" << std::endl;

    // Копируем результаты в изображения
    memcpy(imagePart1.getData(), outputData1.data(), outputData1.size());
    memcpy(imagePart2.getData(), outputData2.data(), outputData2.size());

    // Собираем результат обратно
    std::cout << "Merging processed parts..." << std::endl;

    // Вставляем обработанные части, исключая перекрытие
    image.insertSubImage(imagePart1, 0);

    // Для нижней части исключаем перекрытие
    BMPImage imagePart2NoOverlap = imagePart2.createSubImage(overlap, imagePart2.getHeight());
    image.insertSubImage(imagePart2NoOverlap, midY);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "========================================" << std::endl;
    std::cout << "Total processing time with 2 threads: " << duration.count() << " ms" << std::endl;
}

int main() {

    std::string input_file = "test_3.bmp";
    std::string output_file = "output_3.bmp";
    std::string output_file_CPU = "output_3_CPU.bmp";
    std::string output_file_multigpu = "output_3_multigpu.bmp";

    BMPImage image;
    // Прогрев
    if (!image.load(input_file)) {
        std::cerr << "Error: Failed to load image" << std::endl;
        return 1;
    }
    applyEdgeDetection(image);
    //
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

    std::cout << "Loading image: " << input_file << std::endl;
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

    std::cout << "\nApplying edge detection on multiGPU..." << std::endl;
    applyEdgeDetectionTwoStreamsSeparateImages(image);

    std::cout << "\nSaving result to: " << output_file_multigpu << std::endl;
    if (!image.save(output_file_multigpu)) {
        std::cerr << "Error: Failed to save image" << std::endl;
        return 1;
    }

    std::cout << "Result saved successfully" << std::endl;

    if (!image.load(input_file)) {
        std::cerr << "Error: Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "\nApplying edge detection on multiGPU..." << std::endl;
    applyEdgeDetectionTwoThreads(image);

    std::cout << "\nSaving result to: " << output_file_multigpu << std::endl;
    if (!image.save(output_file_multigpu)) {
        std::cerr << "Error: Failed to save image" << std::endl;
        return 1;
    }

    std::cout << "Result saved successfully" << std::endl;

    return 0;
}