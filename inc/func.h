#pragma once

#include <random>
#include <vector>
#include <functional>
#include "csv.h"

using dataset_t = std::vector<std::vector<double>>;
using datasetPair_t =
    std::pair<dataset_t, dataset_t>;

datasetPair_t createDataset(double rangeMin, double rangeMax,
                            size_t setSize, size_t inputCount,
                            std::mt19937& rng,
                            std::function<double(double)> callback);

datasetPair_t createDataset(std::string const& path,
                            size_t setSize,
                            size_t inputCount,
                            size_t outputCount);
namespace sigmoid {
double function(double x);
double derivative(double x);
}

namespace relu {
double function(double x);
double derivative(double x);
}
