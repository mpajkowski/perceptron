#include "func.h"
#include "propagation.h"
#include "activationFunction.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <functional>

void adjustHelpers(std::vector<double>& inputLayer,
                   std::vector<layer_t>& hiddenLayers,
                   layer_t& outputLayer,
                   std::vector<double>& h1,
                   std::vector<double>& h2)
{
    std::vector<size_t> s;

    s.push_back(inputLayer.size());
    s.push_back(outputLayer.size());

    for (auto& layer : hiddenLayers) {
        s.push_back(layer.size());
    }

    auto maxVal = std::max_element(s.begin(), s.end());
    h1.resize(*maxVal);
    h2.resize(*maxVal);
}

void training(size_t epochs, std::vector<double>& inputLayer,
              std::vector<layer_t>& hiddenLayers,
              layer_t& outputLayer,
              std::vector<double>& h1,
              std::vector<double>& h2,
              std::mt19937& rng)
{
    auto regularTest = [&]() 
    {
        auto t = test(inputLayer, hiddenLayers, outputLayer, h1, h2, rng);
        std::cout << "Good: " << t.first << ", Bad: " << t.second << "\n";
    };
    
    std::uniform_real_distribution<> dist100(0, 100.0);
    std::vector<std::vector<double>> inputLearnSignals;
    std::vector<std::vector<double>> outputLearnSignals;

    for (size_t i = 0; i < 1000; ++i) {
        inputLearnSignals.emplace_back();
        outputLearnSignals.emplace_back();

        for (size_t j = 0; j < INPUT_COUNT; ++j) {
            inputLearnSignals[i].emplace_back(dist100(rng));
            outputLearnSignals[i].emplace_back(std::sqrt(inputLearnSignals[i][j]));
        }
    }

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
            regularTest();
        }
    }
    regularTest();
}

std::pair<int, int> test(std::vector<double>& inputLayer,
                         std::vector<layer_t>& hiddenLayers,
                         layer_t& outputLayer,
                         std::vector<double>& h1,
                         std::vector<double>& h2,
                         std::mt19937& rng)
{
    int positiveAnswers = 0;
    int negativeAnswers = 0;

    std::uniform_real_distribution<> dist100(0, 100.0);
    std::vector<std::vector<double>> inputTestSignals;
    std::vector<std::vector<double>> outputTestSignals;

        for (size_t i = 0; i < 1000; ++i) {
        inputTestSignals.emplace_back();
        outputTestSignals.emplace_back();

        for (size_t j = 0; j < INPUT_COUNT; ++j) {
            inputTestSignals[i].emplace_back(dist100(rng));
            outputTestSignals[i].emplace_back(std::sqrt(inputTestSignals[i][j]));
        }
    }

    for (size_t i = 0; i < inputTestSignals.size(); ++i) {
        inputLayer = inputTestSignals[i];
        h1 = inputLayer;
        forwardPropagation(hiddenLayers, h1, h2);

        for (size_t j = 0; j < outputLayer.size(); ++j) {
            outputLayer[j].setInputs(h2);
            outputLayer[j].updateSum();

            double estimator = outputTestSignals[i][j];
            double response = sigmoid::function(outputLayer[j].getSum()) * 10;
            double error = estimator - response;
            double mse = pow((estimator - response), 2);

            if (mse < 0.25) {
                ++positiveAnswers;
            }
            else
            {
                ++negativeAnswers;
            }
//            std::cout << "Wanted: "<< std::setw(7) << estimator << "\tResponse: " << std::setw(7) << response
//                << "\tAccuracy: " << std::setw(7) << mse << "\n";

        }
    }

    return std::make_pair(positiveAnswers, negativeAnswers);
}
