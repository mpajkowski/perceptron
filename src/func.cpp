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

datasetPair_t createDataset(std::string const& path,
                            size_t setSize,
                            size_t inputCount,
                            size_t outputCount)
{
    size_t totalCount = inputCount + outputCount;
    io::CSVReader<8> in{path};
    double buffer[8];
    dataset_t input(setSize);
    dataset_t output(setSize);
    
    for (size_t i = 0; in.read_row(buffer[0], buffer[1], buffer[2],
                                   buffer[3], buffer[4], buffer[5],
                                   buffer[6], buffer[7]); ++i) {
        for (size_t j = 0; j < inputCount; ++j) {
            input[i].emplace_back(buffer[j]);
        } 

        for (size_t j = inputCount; j < totalCount; ++j) {
            output[i].emplace_back(buffer[j]);
        }
    }

    return std::make_pair(input, output);
}

// Dummy function for memtests
// double std::exp(double) does not like valgrind
double dummy(double x) { return 0; }

double sigmoid::function(double x)
{
    return 1. / (1. + std::exp(-x));
}

double sigmoid::derivative(double x)
{
    return sigmoid::function(x) * (1 - sigmoid::function(x));
}

double relu::function(double x)
{
    return 1 - (x < 0 ? .01 * x : x);
}

double relu::derivative(double x)
{
    return 1 - (x < 0 ? .01 : 1);
}

