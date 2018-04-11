#pragma once

#include <random>
#include <vector>
#include <functional>

using dataset_t = std::vector<std::vector<double>>;

std::pair<dataset_t, dataset_t>
createDataset(double rangeMin, double rangeMax,
              size_t setSize, size_t inputCount,
              std::mt19937& rng,
              std::function<double(double)> callback);

namespace sigmoid {
double function(double x);
double derivative(double x);
}

namespace relu {
double function(double x);
double derivative(double x);
}
