#include "propagation.h"
#include "func.h"

#include <random>
#include <iostream>

std::vector<double> inputLayer;
std::vector<layer_t> hiddenLayers;
layer_t outputLayer;
std::vector<std::vector<double>> inputLearnSignals;
std::vector<std::vector<double>> outputLearnSignals;
std::vector<std::vector<double>> inputTestSignals;
std::vector<std::vector<double>> outputTestSignals;
std::vector<double> h1;
std::vector<double> h2;

int main()
{
    std::mt19937 rng;
    rng.seed(std::random_device{}());

    // Populate layers
    for (size_t i = 0; i < INPUT_COUNT; ++i) {
        inputLayer.push_back(0.0);
        outputLayer.emplace_back(NEURON_COUNT, true, rng);
    }

    layer_t u1;
    for (size_t i = 0; i < NEURON_COUNT; ++i) {
        u1.emplace_back(NEURON_COUNT, true, rng);
    }
    hiddenLayers.push_back(u1);

    // Populate data
    std::uniform_real_distribution<> dist100(0, 100.0);
    std::uniform_real_distribution<> dist200(0, 200.0);
    for (size_t i = 0; i < 1000; ++i) {
        inputTestSignals.emplace_back();
        outputTestSignals.emplace_back();
        inputLearnSignals.emplace_back();
        outputLearnSignals.emplace_back();

        for (size_t j = 0; j < INPUT_COUNT; ++j) {
            inputLearnSignals[i].emplace_back(dist100(rng));
            outputLearnSignals[i].emplace_back(std::sqrt(inputLearnSignals[i][j]));
            inputTestSignals[i].emplace_back(dist100(rng));
            outputTestSignals[i].emplace_back(std::sqrt(inputTestSignals[i][j]));
        }
    }

    h1.resize(100);
    h2.resize(100);

    training(4000, inputLayer, hiddenLayers, outputLayer,inputLearnSignals,
             outputLearnSignals, h1, h2);

    auto testResult = test(inputLayer, hiddenLayers, outputLayer,
                           inputTestSignals, outputTestSignals,
                           h1, h2);

    std::cout << "Good: " << testResult.first << ", Bad: " << testResult.second << std::endl;
}
