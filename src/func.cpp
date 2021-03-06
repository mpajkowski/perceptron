#include "func.h"
#include <cmath>
#include <algorithm>
#include <random>


datasetPair_t createDataset(double rangeMin, double rangeMax,
                            size_t setSize, size_t inputCount,
                            std::mt19937& rng,
                            std::function<double(double)> callback)
{
    dataset_t input(setSize);
    dataset_t output(setSize);

    std::uniform_real_distribution<> dist(rangeMin, rangeMax);

    for (size_t i = 0; i < setSize; ++i) {
        for (size_t j = 0; j < inputCount; ++j) {
            input[i].emplace_back(dist(rng));
            output[i].emplace_back(callback(input[i][j]));
        }
    }

    return std::make_pair(input, output);
}

double sigmoid::function(double x)
{
    return 1. / (1. + std::exp(-x));
}

