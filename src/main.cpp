#include "propagation.h"
#include "func.h"

#include <random>
#include <iostream>

int main()
{
    std::vector<double> inputLayer;
    std::vector<layer_t> hiddenLayers;
    layer_t outputLayer;

    std::vector<std::vector<double>> inputLearnSignals;
    std::vector<std::vector<double>> outputLearnSignals;
    std::vector<std::vector<double>> inputTestSignals;
    std::vector<std::vector<double>> outputTestSignals;

    std::vector<double> h1;
    std::vector<double> h2;

    std::mt19937 rng;
    rng.seed(std::random_device{}());

    // Populate layers
    // Maybe that't time to write sume nice Net class, huh TODO
    // 1  x  60 x  50  x  1
    //       *     *
    // *     *     *      *    x 2200
    //       *     *
    //       *     *
    //       *     *
    //       *
    // i  hl1 hl2  o

    int hl1Size = 60;
    int hl2Size = 50;

    for (size_t i = 0; i < INPUT_COUNT; ++i) {
        inputLayer.push_back(0.0);
        outputLayer.emplace_back(hl2Size, true, rng);
    }

    layer_t hl1, hl2;
    for (size_t i = 0; i < hl1Size; ++i) {
        hl1.emplace_back(INPUT_COUNT, true, rng);
    }

    for (size_t i = 0; i < hl2Size; ++i) {
        hl2.emplace_back(hl1Size, true, rng);
    }

    hiddenLayers.push_back(hl1);
    hiddenLayers.push_back(hl2);

    adjustHelpers(inputLayer, hiddenLayers, outputLayer, h1, h2);
    training(2200, inputLayer, hiddenLayers, outputLayer, h1, h2, rng);
}
