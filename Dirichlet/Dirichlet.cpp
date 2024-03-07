#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

class wavefront_approximator {
    const int size;
    const double h;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> f;

    const int block_size = 32;
    const double eps = 0.01;
    double (*fun)(double, double);
    const int nb;
    
    inline double process_block(int align_i, int align_j) {
        int start_block_i = 1 + align_i * block_size;
        int end_block_i = std::min(start_block_i + block_size, size);
        int start_block_j = 1 + align_j * block_size;
        int end_block_j = std::min(start_block_j + block_size, size);

        double dm = 0;
        for (int i = start_block_i; i < end_block_i; i++) {
            for (int j = start_block_j; j < end_block_j; j++) {
                double temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                double d = temp - u[i][j];
                if (dm < d)
                    dm = d;
            }
        }
        return dm;
    }

public:
    wavefront_approximator(int size, double (*fun_g)(double, double), double (*fun_f)(double, double)) : size(size), h(1.0 / (size + 1)), fun(fun_g), nb(size / block_size + (size % block_size == 0)) {
        u = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));
        f = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));

        for (int i = 1; i < size + 1; i++) {
            for (int j = 1; j < size + 1; j++) {
                f[i][j] = fun_f(i * h, j * h);
            }
        }

        for (int i = 0; i < size + 2; i++) {
            u[i][0] = fun_g(i * h, 0);
            u[i][size + 1] = fun_g(i * h, (size + 1) * h);
            u[0][i] = fun_g(0, i * h);
            u[size + 1][i] = fun_g((size + 1) * h, i * h);
        }
    }

    void process_net() {
        double dmax = 0;
        std::vector<double> dm(nb, 0);
        do {
            for (int nx = 0; nx < nb; nx++) {
#pragma omp parallel for shared(nx, dm)
                for (int i = 0; i <= nx; i++) {
                    int j = nx - i;
                    double d = process_block(i, j);
                    if (dm[i] < d)
                        dm[i] = d;
                }
            }
            for (int nx = nb - 2; nx >= 0; nx--) {
#pragma omp parallel for shared(nx, dm)
                for (int i = 1; i <= nx; i++) {
                    int j = 2 * (nb - 1) - nx - i;
                    double d = process_block(i, j);
                    if (dm[i] < d)
                        dm[i] = d;
                }
            }
            for (int i = 0; i < nb; i++) {
                if (dmax < dm[i])
                    dmax = dm[i];
            }
        } while (dmax > eps);
    }
};

int main() {
    const int threads_num = 4;
    omp_set_num_threads(threads_num);

    auto fun = [](double x, double y) { return x * sin(x) + cos(y) / y; };
    auto fun_d = [](double x, double y) { return 2 * cos(x) - x * sin(x) - 2 * sin(y) - y * cos(y); };
    wavefront_approximator net(100, fun, fun_d);

    auto start_time = omp_get_wtime();
    net.process_net();
    auto end_time = omp_get_wtime();

    std::cout << end_time - start_time << std::endl;
}
