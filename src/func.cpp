#include "func.h"
#include "propagation.h"

#include <cmath>
#include <iostream>
#include <iomanip>

void training(size_t epochs, std::vector<double>& inputLayer,
              std::vector<layer_t>& hiddenLayers,
              layer_t& outputLayer,
              std::vector<std::vector<double>>& inputLearnSignals,
              std::vector<std::vector<double>>& outputLearnSignals,
              std::vector<double>& h1,
              std::vector<double>& h2)
{
    while (epochs --> 0) {
        for (size_t i = 0; i < inputLearnSignals.size(); ++i) {
            inputLayer = inputLearnSignals[i];
            h1 = inputLayer;

            forwardPropagation(hiddenLayers, h1, h2);
            backpropOutput(outputLayer, h2, outputLearnSignals[i]);
            backpropHidden(hiddenLayers, outputLayer);
            updateNeurons(hiddenLayers, outputLayer);
        }

        if (!(epochs % 1000)) {
            for (auto& layer : hiddenLayers) {
                for (auto& neuron : layer) {
                    neuron.increaseLearnFactor();
                }
            }
        }
        std::cout << epochs << "\n";
    }
}

std::pair<int, int> test(std::vector<double>& inputLayer,
                         std::vector<layer_t>& hiddenLayers,
                         layer_t& outputLayer,
                         std::vector<std::vector<double>>& inputTestSignals,
                         std::vector<std::vector<double>>& outputTestSignals,
                         std::vector<double>& h1,
                         std::vector<double>& h2)
{
    int positiveAnswers = 0;
    int negativeAnswers = 0;

    for (size_t i = 0; i < inputTestSignals.size(); ++i) {
        for (size_t j = 0; j < inputLayer.size(); ++j) {
            inputLayer[j] = inputTestSignals[i][j];
        }

        h1 = inputLayer;
        forwardPropagation(hiddenLayers, h1, h2);

        for (size_t j = 0; j < outputLayer.size(); ++j) {
            outputLayer[j].setInputs(h2);
            outputLayer[j].getSum();
            double response = outputLayer[i].activate() * 10;
            if (std::abs(outputTestSignals[i][j]
                         - response) <= 0.5) {
                ++positiveAnswers;
            }
            else
            {
                ++negativeAnswers;
            }
            std::cout << "Wanted: "<< std::setw(7) << outputTestSignals[i][j] << "   Response: " << std::setw(7) << response
                << "   Diff: " << std::setw(7) << std::abs(outputTestSignals[i][j] - response)
                << "\tAccuracy: " << std::setw(7) << (response / outputTestSignals[i][j]) << "\n";
        }
    }

    auto ret = std::make_pair(positiveAnswers, negativeAnswers);
    return ret;
}