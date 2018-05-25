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

std::tuple<datasetPair_t, datasetPair_t, datasetPair_t>
processIris(std::string const& path, std::mt19937& rng)
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
    dataset_t inputValidate;
    dataset_t outputValidate;
    std::uniform_real_distribution<> dist{0,1};

    while (in.read_row(buffer[0], buffer[1], buffer[2],
                       buffer[3], buffer[4], buffer[5],
                       buffer[6])) {
        double randNum = dist(rng);

        dataset_t* input;
        dataset_t* output;

        if (randNum > .0 && randNum < .6) {
            // learn
            input = &inputLearn;
            output = &outputLearn;
        } else if (randNum > .6 && randNum < .9) {
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

        for (size_t j = 0; j < 4; ++j) {
            input->back().emplace_back(buffer[j]);
        }

        for (size_t j = 4; j < 7; ++j) {
            output->back().emplace_back(buffer[j]);
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

