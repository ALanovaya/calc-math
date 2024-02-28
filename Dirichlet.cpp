#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>

class wavefront_approximator {
public:
    int size;
    double h;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> f;

private:
    const int block_size = 32;
    const double eps = 0.01;
    double (*fun_f)(double, double);
    double (*fun_g)(double, double);

public:
    wavefront_approximator(int size, double (*fun_f)(double, double), double (*fun_g)(double, double)) : size(size), h(1.0 / (size + 1)) {
        u = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));
        f = std::vector<std::vector<double>>(size + 2, std::vector<double>(size + 2, 0.0));
        this->fun_f = fun_f;
        this->fun_g = fun_g;

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

    double process_block(int align_i, int align_j) {
        int start_block_i = 1 + align_i * block_size;
        int end_block_i = std::min(start_block_i + block_size, size);
        int start_block_j = 1 + align_j * block_size;
        int end_block_j = std::min(start_block_j + block_size, size);

        double dm = 0;
        for (int i = start_block_i; i < end_block_i; i++) {
            for (int j = start_block_j; j < end_block_j; j++) {
                double temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                double d = fabs(temp - u[i][j]);
                if (dm < d)
                    dm = d;
            }
        }
        return dm;
    }

    void process_net() {
        int nb = size / block_size;
        if (block_size * nb != size)
            nb += 1;
        double dmax = 0;
        std::vector<double> dm = std::vector<double>(nb, 0.0);

        do {
            dmax = 0;
            for (int nx = 0; nx < nb; nx++) {
                dm[nx] = 0;
                
#pragma omp parallel for shared(nx, dm)
                for (int i = 0; i < nx + 1; i++) {
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

            for (int i = 0; i < nb; i++)
                if (dmax < dm[i])
                    dmax = dm[i];
        } while (dmax > eps);
    }
};

int main() {
    const int threads_num = 10;
    omp_set_num_threads(threads_num);

    auto fun_g = [](double x, double y) { return 3 * pow(x, 5) + 8 * pow(y, 4); };
    auto fun_f = [](double x, double y) { return 60 * pow(x, 3) + 96 * pow(y, 2); };
    wavefront_approximator net(1000, fun_f, fun_g);

    auto start_time = std::chrono::system_clock::now();
    net.process_net();
    auto mills = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();

    std::cout << mills;
}
