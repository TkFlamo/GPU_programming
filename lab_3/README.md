# Лабораторная работа 3

Работа выполнялась в среде Visual Studio 2022 на локальном компьютере с ОС Windows. 

В репозитории представленна папка с проектом, основной код в файле kernel.cu (GPU_lab_3/GPU_lab_3/kernel.cu). Там же представленны тестовые картинки и результаты работы.

### Описание работы

Тестировлась скорость обработки изображений оператором Собеля на CPU, GPU, GPU в двух потоках. Разрезка и склейка изображения производилась с учетом пересечения, для корректной работы ядра.

#### Размер картинки 3840x2160

CPU:

CPU time: 863 ms

Processing time: 872 ms


GPU:

Kernel time: 3.32883

Processing time: 19 ms


GPU 2 stream:

Kernel time: 2.22573, 2.2544

Parallel processing time: 15 ms

Processing time: 51 ms



#### Размер картинки 7680x4320

CPU:

CPU time: 3147 ms

Processing time: 3185 ms


GPU:

Kernel time: 13.6681

Processing time: 71 ms


GPU 2 stream:

Kernel time: 94.1145, 155.511

Parallel processing time: 244 ms

Processing time: 321 ms