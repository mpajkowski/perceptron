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

    return {input, output};
}

std::tuple<dataset_t, dataset_t, dataset_t, dataset_t>
processIris(std::string const& path)
{
    /* MAGIC NUMBERS
     *
     * 4 attributes + 3 possible outputs = 7 rows total
     * 150 - size of dataset
     * 
     *
     */
    io::CSVReader<7> in{path};
    double buffer[7];
    dataset_t inputLearn;
    dataset_t outputLearn;
    dataset_t inputTest;
    dataset_t outputTest;

    for (size_t i = 0; in.read_row(buffer[0], buffer[1], buffer[2],
                                   buffer[3], buffer[4], buffer[5],
                                   buffer[6]); ++i) {
        auto& input  = (i + 1) % 3 ? inputLearn  : inputTest;
        auto& output = (i + 1) % 3 ? outputLearn : outputTest;

        input.emplace_back();
        output.emplace_back();

        for (size_t j = 0; j < 4; ++j) {
            input.back().emplace_back(buffer[j]);
        }

        for (size_t j = 4; j < 7; ++j) {
            output.back().emplace_back(buffer[j]);
        }
    }

    return {inputLearn, outputLearn, inputTest, outputTest};
}

double sigmoid::function(double x)
{
    return 1. / (1. + std::exp(-x));
}

