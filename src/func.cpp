#include "func.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

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

std::tuple<datasetPair_t, datasetPair_t, datasetPair_t>
processIris(std::string const& path, std::mt19937& rng)
{
    io::CSVReader<7> in{path};
    std::array<double, 7> buffer;

    // fetch data
    dataset_t fetchedRows;

    while (in.read_row(buffer[0], buffer[1], buffer[2],
                       buffer[3], buffer[4], buffer[5],
                       buffer[6])) {
        fetchedRows.emplace_back();

        for (size_t i = 0; i < 7; ++i) {
            fetchedRows.back().emplace_back(buffer[i]);
        }
    }

    // normalize data
    // first, find min and max
    std::array<double, 4> maxValues;
    std::array<double, 4> minValues;

    for (auto& x : maxValues) {
        x = 0;
    }

    for (auto& x : minValues) {
        x = std::numeric_limits<double>::max();
    }

    for (auto const& row : fetchedRows) {
        for (size_t i = 0; i < maxValues.size(); ++i) {
            if (row[i] < minValues[i]) {
                minValues[i] = row[i];
            }

            if (row[i] > maxValues[i]) {
                maxValues[i] = row[i];
            }
        }
    }

    // normalize
    for (auto& row : fetchedRows) {
        for (size_t i = 0; i < 4; ++i) {
            row[i] = (row[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    } 
    
    // split into input, output, validate
    dataset_t inputLearn;
    dataset_t outputLearn;
    dataset_t inputTest;
    dataset_t outputTest;
    dataset_t inputValidate;
    dataset_t outputValidate;
    std::uniform_real_distribution<> dist01{0,1};

    for (auto const& row : fetchedRows) {
        double randNum = dist01(rng);
        dataset_t* input;
        dataset_t* output;

        if (randNum > 0.0 && randNum < .6) {
            // learn
            input = &inputLearn;
            output = &outputLearn;
        } else if (randNum > .6 && randNum < .8) {
            // test
            input = &inputTest;
            output = &outputTest;
        } else {
            // validate
            input = &inputValidate;
            output = &outputValidate;
        }

        input->emplace_back();
        output->emplace_back();

        for (size_t i = 0; i < 4; ++i) {
            input->back().emplace_back(row[i]);
        }

        for (size_t i = 4; i < 7; ++i) {
            output->back().emplace_back(row[i]);
        }
    }

    return {
        { inputLearn, outputLearn },
        { inputTest, outputTest },
        { inputValidate, outputValidate }
    };
}

double sigmoid::function(double x)
{
    return 1. / (1. + std::exp(-x));
}

