#include <iostream>
#include <vector>

#include "qwetensor.hpp"

void print_tensor(qwe::Tensor<> &T)
{
    int k = 0;

    /// Вывод тензора по строкам с помощью итератора
    for (qwe::Tensor<>::Iterator i = T.hor_begin(); i != T.hor_end(); i++)
    {
        std::cout << *i << " ";

        k++;
        if (k == qwe::default_dim)
        {
            std::cout << std::endl;
            k = 0;
        }
    }
}

int main() {
    /// Можно инициализировать тензора с помощью вектора векторов
    /// чисел типа vcoord_t.
    qwe::Tensor<> T(std::vector<std::vector<qwe::vcoord_t> >(3, std::vector<qwe::vcoord_t>(3, 5)));

    /// Можно создать пустой тензор и заполнить его с помощью операции прямого доступа
    qwe::Tensor<> T2;
    T2[1][1] = 0.312;
    T2[0][2] = 2.00004;

    /// А потом посмотреть, что получилось
    std::cout << T2[0][1] << std::endl;

    /// Сложение
    qwe::Tensor<> Ta = T + T2;
    print_tensor(Ta);

    /// Умножение на число
    qwe::Tensor<> Ts = T * 0.3213;
    print_tensor(Ts);

    /// Перемножение тензоров
    qwe::Tensor<> Tm = T * Ta;
    print_tensor(Tm);

    /// Двойное скалярное произведение
    std::cout << (T^Ta) << std::endl;
    
    /// Умножение тензора на вектор
    std::vector<qwe::vcoord_t> v(3, 7);
    v = Tm * v;

    /// Транспонирование
    qwe::Tensor<> Tt = Tm.transpose();
    print_tensor(Tt);

    /// Шаблоны используются неспроста. Так мы добиваемся статической
    /// проверки соответствия размерностей.
    qwe::Tensor<2> T2d;
    /// Умножить T2d на T не получится.
}
