# Лабораторная работа 1

Работа выполнялась в среде Visual Studio 2022 на локальном компьютере с ОС Windows. 

В репозитории представленна папка с проектом, основной код в файле kernel.cu (GPU_lab_1/GPU_lab_1/kernel.cu).

### Характеристики видеокарты

Device name : NVIDIA GeForce RTX 4070
Total global memory : 12281 MB
Shared memory per block : 49152
Registers per block : 65536
Warp size : 32
Max threads per block : 1024
Max threads dimensions : x = 1024, y = 1024, z = 64
Max grid size: x = 2147483647, y = 65535, z = 65535

### Описание работы

Оценивалась скорость заполнения большого массива (~ 10^9 элементов) результатами функции sin на GPU и точность вычислений. 

Тестировалось несколько функций sin, sinf, __sinf, работа с float и double.

float sin:
Time: 6.27306
Mean error: 7.59934e-08

float sinf:
Time: 5.15206
Mean error: 7.59934e-08

float __sinf:
Time: 3.56477
Mean error: 1.14845e-07

double sin:
Time: 99.8172
Mean error: 9.31932e-18

double sinf:
Time: 160.786
Mean error: 4.67949e-08

double __sinf:
Time: 51.3484
Mean error: 1.30149e-07

### Выводы

Наибольшую точность (17-18 цифр) показала ф-ция sin при работе с double. 

sinf и __sinf снижают точность до уровня float (7-8 цифр), причем работая с double тратят время на конвертацию.  

__sinf как встроенная оптимизированная ф-ция быстрее (особенно с double), но показывает немного более низкую точность. 