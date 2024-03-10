#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>

class wavefront_approximator {
    const int size;
    const double h;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> f;

    const int block_size = 16;
    const double eps = 10e-14;
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
                u[i][j] = 0.25 * fabs(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                double d = fabs(temp - u[i][j]);
                if (dm < d)
                    dm = d;
            }
        }
        return dm;
    }

public:
    wavefront_approximator(int size, double (*fun_g)(double, double), double (*fun_f)(double, double)) : size(size), h(1.0 / (size + 1)), fun(fun_g), nb(size / block_size + (size % block_size == 0)) {
        f = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));
        u = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));
    
        double sum = 0.0;

        for (int i = 0; i < size + 2; i++) {
            u[i][0] = fun_g(i * h, 0);
            u[i][size + 1] = fun_g(i * h, (size + 1) * h);
            u[0][i] = fun_g(0, i * h);
            u[size + 1][i] = fun_g((size + 1) * h, i * h);
            sum += fun_g(i * h, 0) + fun_g(i * h, (size + 1) * h) + fun_g(0, i * h) + fun_g((size + 1) * h, i * h);
        }

        sum = (sum - fun_g(0, 0) - fun_g(0, (size + 1) * h) - fun_g((size + 1) * h, 0) - fun_g((size + 1) * h, (size + 1) * h)) / (4 * (size + 1));

        for (int i = 1; i < size + 1; i++) {
            for (int j = 1; j < size + 1; j++) {
                f[i][j] = fun_f(i * h, j * h);
                u[i][j] = sum;
            }
        }
    }

    void process_net() {
        double dmax = 0;
        std::vector<double> dm(nb, 0);
        do {
            dmax = 0;
            for (int nx = 0; nx < nb; nx++) {
                dm[nx] = 0;
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
                for (int i = 1; i < nx + 1; i++) {
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

    double test_results() {
        const double zero  = 10e-15;
        double sum_error = 0.0;
        for (int i = 1; i <= size; i++) {
            for (int j = 1; j <= size; j++) {
                if (fabs(fun(i * h, j * h)) > zero)
                    sum_error += fabs((u[i][j] - fun(i * h, j * h))) / fabs(fun(i * h, j * h));  
            }
        }
        return sum_error / (size * size);
    }
};

int main() {
    const int threads_num = 6;
    omp_set_num_threads(threads_num);

    auto fun = [](double x, double y) { return exp(-x * pow(y, 3)) ; };
    auto fun_d = [](double x, double y) { return pow(y, 6) * exp(-x * pow(y, 3)) + 3 * x * exp(-x * pow(y, 3)) * (3 * x * pow(y, 3) - 2); };

    std::vector<int> test_net_sizes{20, 40, 80, 160, 320};

    for (auto it: test_net_sizes) {
        wavefront_approximator net(it, fun, fun_d);
        auto start_time = omp_get_wtime();
        net.process_net();
        auto end_time = omp_get_wtime();

        std::cout << end_time - start_time << std::endl;
        std::cout << net.test_results() << std::endl;
    }
}
