# Сравнение работы алгоритмов 
Для сравнительного анализы были выбраны:
+ Стандартный алгоритм из библиотеки numpy 
+ Примитивный алгоритм -- реализация степенного метода
+ [Block SVD power method](https://www.degruyter.com/document/doi/10.1515/jisys-2018-0034/html)

Запуск проводился на машине с ОС `Ubuntu 20.04.5 LTS x86_64`, версией python `3.12.1`, версией библиотеки numpy `1.26.4`, на процессоре ``11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz``.

Помимо коэффициента сжатия можно также регулировать время работы алгоритма, зафиксируем коэффициент сжатия, пусть он равен 2.

Стандартный алгоритм на картинке 900х900 пикселей в среднем за 10 запусков работает за 1.62 секунды. Попробуем запустить самописные алгоритмы с таким же ограничение по времени.


| Numpy                            | Power simple                            | Block power                            |
|----------------------------------|-----------------------------------------|----------------------------------------|
| ![](img/numpy_2_ogr.bmp) | ![](img/simple_2_ogr.bmp) | ![](img/advanced_2_ogr.bmp) |

Как можно заметить, качество результирующих изображений самописных алгоритмов гораздо хуже по сравнения со стандартным.

Попробуем еще увеличить время для блочного алгоритма.
| 1000 ms                          | 3000 ms                           | 8000 ms                           |
|----------------------------------|-----------------------------------------|----------------------------------------|
| ![](img/advanced_2_ogr.bmp) | ![](img/adv_t_1.bmp) | ![](img/adv_t_2.bmp) |

Качество становится еще лучше, это вполне логичный результат, так как повышается точность с подсчетом большего количества сингулярных чисел.

Теперь зафиксируем количество времени на обработку канала = 1000ms для самописных алгоритмов, и рассмотрим сжатие в 5 раз, они справились почти идентично.

| Numpy                            | Power simple                            | Block power                            |
|----------------------------------|-----------------------------------------|----------------------------------------|
| ![](img/num_5.bmp) | ![](img/simple_5_new.bmp) | ![](img/adv_5.bmp) |
 
# Формат промежуточного представления

Файл содержит данные о сингулярном разложении $A=U\Sigma V^T$ для каждого цветового канала в изображении. Формат файла описывается следующей структурой:

| Смещение (шестнадцатеричное) | Размер (байты) | Описание | 
|----------------------------------|-----------------------------------------|----------------------------------------|
| 00 | 4 | Заголовок формата | 
| 04 | 4 | Высота изображения n (целое число) |
| 08 | 4 | Ширина изображения m (целое число) |
| 0C | 4 | Количество вычисленных сингулярных значений k (целое число) | 
| 10 | $3 * 4 * k * (n + m + 1)$ | Информация о трех цветовых составляющих. Записаны последовательно данные каналов без разделителей: матрица U, k сингулярных чисел, матрица $V^T$. Все значения представлены в формате с плавающей запятой (float32). |
