#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace HeatEquation {
double lambda;

// Function 'f' for the heat equation source term
double f(double x) {
  return 2 * lambda * sin(sqrt(lambda) * x);
}

// Function 'p', assumed to be constant 1 in this context
double p(double) {
  return 1;
}

// Function 'q', dependent on lambda1
double q(double x) {
  return lambda;
}

// Exact solution of the differential equation
double analyticalSolution(double x) {
  return sin(sqrt(lambda) * x);
}

} // namespace HeatEquation

// Calculate the value of the j-th basis function at point x
double phi_j(int j, const std::vector<double>& xs, double x) {
  int n = xs.size();
  if (j < 0 || j >= n) {
    return 0;
  } else if (j == 0) {
    if (x <= xs[1])
      return (xs[1] - x) / (xs[1] - xs[0]);
  } else if (j == n - 1) {
    if (x >= xs[n - 2])
      return (x - xs[n - 2]) / (xs[n - 1] - xs[n - 2]);
  } else {
    if (xs[j - 1] <= x && x <= xs[j])
      return (x - xs[j - 1]) / (xs[j] - xs[j - 1]);
    else if (xs[j] <= x && x <= xs[j + 1])
      return (xs[j + 1] - x) / (xs[j + 1] - xs[j]);
  }
  return 0;
}

// Simple trapezoidal rule for numerical integration
double trapezoidalRule(const std::function<double(double)>& func, double a, double b, int n = 100) {
    double h = (b - a) / n;
    double sum = 0.5 * (func(a) + func(b));

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += func(x);
    }

    return sum * h;
}

// Helper function to calculate h_i and h_i+1
std::pair<double, double> calculateH(const std::vector<double>& xs, int j) {
    return {xs[j] - xs[j - 1], xs[j + 1] - xs[j]};
}

// Helper function to calculate a[i][j] elements for the tridiagonal matrix
double calculateAij(const std::vector<double>& xs, double h1, double h2, int j, bool isDiagonal) {
    if (isDiagonal) {
        return (trapezoidalRule([&](double x) { return HeatEquation::p(x) + HeatEquation::q(x) * std::pow(x - xs[j - 1], 2); }, xs[j - 1], xs[j]) / (h1 * h1)) +
               (trapezoidalRule([&](double x) { return HeatEquation::p(x) + HeatEquation::q(x) * std::pow(xs[j + 1] - x, 2); }, xs[j], xs[j + 1]) / (h2 * h2));
    } else {
        return trapezoidalRule([&](double x) { return -HeatEquation::p(x) + HeatEquation::q(x) * (x - xs[j]) * (xs[j + 1] - x); }, xs[j], xs[j + 1]) / (h2 * h2);
    }
}

// Calculate the tridiagonal matrix elements
std::vector<std::vector<double>> calculateMatrix(const std::vector<double>& xs) {
    int n = xs.size();
    std::vector<std::vector<double>> a(3, std::vector<double>(n, 0.0));

    // Dirichlet boundary conditions
    a[1][0] = 1;
    a[1][n - 1] = 1;

    for (int j = 1; j < n - 1; ++j) {
        auto [h1, h2] = calculateH(xs, j);
        a[0][j] = calculateAij(xs, h1, h2, j, false);
        a[1][j] = calculateAij(xs, h1, h2, j, true);
        a[2][j] = calculateAij(xs, h1, h2, j + 1, false);
    }
    return a;
}

// Calculate right side of equation 8.68
std::vector<double> computeRHS(const std::vector<double>& points) {
    size_t n = points.size();
    std::vector<double> rhsValues(n, 0.0);
    for (size_t j = 1; j < n - 1; ++j) {
        auto integrand = [&](double x) {
            return HeatEquation::f(x) * phi_j(j, points, x);
        };
        rhsValues[j] = trapezoidalRule(integrand, points[j - 1], points[j + 1]);
    }
    return rhsValues;
}

// Helper for forward elimination in Thomas algorithm
void forwardElimination(std::vector<double>& main, std::vector<double>& lower, std::vector<double>& upper, std::vector<double>& d) {
    size_t n = d.size();
    for (size_t j = 1; j < n; ++j) {
        double factor = lower[j] / main[j - 1];
        main[j] -= factor * upper[j - 1];
        d[j] -= factor * d[j - 1];
    }
}

// Helper for backward substitution in Thomas algorithm
void backwardSubstitution(const std::vector<double>& main, const std::vector<double>& upper, std::vector<double>& d, std::vector<double>& y) {
    size_t n = d.size();
    y[n - 1] = d[n - 1] / main[n - 1];
    for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
        y[j] = (d[j] - upper[j] * y[j + 1]) / main[j];
    }
}

// New version of thomas with some parts abstracted into functions
std::vector<double> solveTridiagonal(const std::vector<std::vector<double>>& triMatrix, std::vector<double>& dVec) {
    size_t n = dVec.size();
    std::vector<double> lowerDiagonal = triMatrix[0];
    std::vector<double> mainDiagonal = triMatrix[1];
    std::vector<double> upperDiagonal = triMatrix[2];
    std::vector<double> solution(n, 0.0);

    forwardElimination(mainDiagonal, lowerDiagonal, upperDiagonal, dVec);
    backwardSubstitution(mainDiagonal, upperDiagonal, dVec, solution);

    return solution;
}

// New version of eval_f_approx with a different control structure
double approximateFunction(double x, const std::vector<double>& points, const std::vector<double>& solutions) {
    size_t n = points.size();
    double interval = points[1] - points[0];
    size_t index = static_cast<size_t>(std::floor(x / interval));

    if (index >= n - 1) {
        return solutions.back();
    } else {
        double basisLeft = phi_j(index, points, x);
        double basisRight = phi_j(index + 1, points, x);
        return solutions[index] * basisLeft + solutions[index + 1] * basisRight;
    }
}

double calculate_error(const std::vector<double>& nodes, const std::vector<double>& solutions, double boundary) {
    int points = nodes.size();
    std::vector<double> refined_nodes;
    double first = nodes.front();
    double last = nodes.back();
    double increment = (last - first) / (points * 10 - 1);

    // Generate refined nodes
    for (int i = 0; i < points * 10; ++i) {
        refined_nodes.push_back(first + i * increment);
    }

    auto squared_norm = [](const std::vector<double>& vec) {
        return std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    };

    // Calculate norms and errors
    double norm = std::sqrt(trapezoidalRule([&](double x) { return std::pow(HeatEquation::f(x), 2); }, first, last));
    double delta_x_squared = std::pow(nodes[1] - nodes[0], 2);

    std::vector<double> exact_values;
    std::vector<double> approx_values;
    std::transform(refined_nodes.begin(), refined_nodes.end(), std::back_inserter(exact_values), HeatEquation::analyticalSolution);
    std::transform(refined_nodes.begin(), refined_nodes.end(), std::back_inserter(approx_values), [&](double x) {
        return approximateFunction(x, nodes, solutions);
    });

    double error_sum = squared_norm(exact_values) - 2 * std::inner_product(exact_values.begin(), exact_values.end(), approx_values.begin(), 0.0) + squared_norm(approx_values);
    double error = std::sqrt(error_sum);

    // Constraint check
    double constraint = std::sqrt(1 + std::pow(M_PI, 2) / 4) * 0.5;
    double threshold = std::pow((HeatEquation::lambda * boundary / 4 + 1) * constraint, 2) * norm * delta_x_squared;
    if (error > threshold) {
        std::cerr << "Error threshold exceeded." << std::endl;
    }

    return error;
}

int main() {
    std::vector<int> mesh_sizes = {10, 30, 90};
    std::vector<double> boundaries = {M_PI, 2 * M_PI};
    double lower_bound = 0;

    for (auto mesh : mesh_sizes) {
        for (auto boundary : boundaries) {
            HeatEquation::lambda = std::pow(M_PI / boundary, 2);
            std::vector<double> mesh_points(mesh);
            double mesh_step = (boundary - lower_bound) / (mesh - 1);
            std::generate(mesh_points.begin(), mesh_points.end(), [n = 0, lower_bound, mesh_step]() mutable {
                return lower_bound + n++ * mesh_step;
            });

            auto matrix = calculateMatrix(mesh_points);
            auto rhs = computeRHS(mesh_points);
            auto y = solveTridiagonal(matrix, rhs);

            double error_score = calculate_error(mesh_points, y, boundary);

            std::cout << "Mesh: " << mesh << "  Boundary: " << boundary << "   Error: " << error_score << std::endl;
        }
    }

    return 0;
}
