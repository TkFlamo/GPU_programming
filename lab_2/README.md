# Лабораторная работа 2

Работа выполнялась в среде Visual Studio 2022 на локальном компьютере с ОС Windows. 

В репозитории представленна папка с проектом, основной код в файле kernel.cu (GPU_lab_2/GPU_lab_2/kernel.cu). Там же представленны тестовые картинки и результаты работы.

### Описание работы

Тестировлась скорость обработки изображений оператором Собеля на CPU, GPU, GPU с использованием разделяемой памяти.

#### Размер картинки 3840x2160

CPU:

CPU time: 836 ms

Processing time: 843 ms


GPU:

Kernel time: 3.45376

Processing time: 141 ms


GPU shared:

Kernel time: 3.98099

Processing time: 19 ms


#### Размер картинки 800x800

CPU:

CPU time: 57 ms

Processing time: 60 ms


GPU:

Kernel time: 1.42624

Processing time: 127 ms


GPU shared:

Kernel time: 0.440192

Processing time: 4 ms


#### Размер картинки 7680x4320

CPU:

CPU time: 3057 ms

Processing time: 3090 ms


GPU:

Kernel time: 11.0572

Processing time: 204 ms


GPU shared:

Kernel time: 14.0416

Processing time: 100 ms