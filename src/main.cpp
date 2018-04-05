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
        outputLayer.emplace_back(INPUT_COUNT, true, rng);
    }

    for (size_t i = 0; i < HIDDEN_LAYER_COUNT; ++i) {
        hiddenLayers.emplace_back();
        for (size_t j = 0; j< NEURON_COUNT; ++j) {
            hiddenLayers[i].emplace_back(INPUT_COUNT, true, rng);
        }
    }

    // Populate data
    std::uniform_real_distribution<> dist100(0, 100.0);
    for (size_t i = 0; i < INPUT_COUNT; ++i) {
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

    h1.resize(NEURON_COUNT);
    h2.resize(NEURON_COUNT);

    training(100000, inputLayer, hiddenLayers, outputLayer,inputLearnSignals,
             outputLearnSignals, h1, h2);

    auto testResult = test(inputLayer, hiddenLayers, outputLayer,
                           inputTestSignals, outputTestSignals,
                           h1, h2);

    std::cout << "Good: " << testResult.first << ", Bad: " << testResult.second << std::endl;
}