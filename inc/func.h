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

template<size_t csvCol>
datasetPair_t createDataset(std::string const& path,
                            size_t setSize,
                            size_t inputCount,
                            size_t outputCount)
{
    size_t totalCount = inputCount + outputCount;
    io::CSVReader<csvCol> in{path};
    double buffer[csvCol];
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

namespace sigmoid {
double function(double x);
}

