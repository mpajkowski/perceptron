#include "func.h"
#include "propagation.h"
#include "activationFunction.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>

std::pair<dataset_t, dataset_t>
createDataset(double rangeMin, double rangeMax,
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
    std::fill(h1.begin(), h1.end(), .0);
    std::fill(h2.begin(), h2.end(), .0);
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
    
    auto [inputLearnSignals, outputLearnSignals] = createDataset(0., 100., 1000, INPUT_COUNT, rng, sqrt);

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
                    neuron.learnF += .001;
                }
            }
            regularTest();
        }
    }
    std::cout << "Learning ended!\n";
    regularTest();
    regularTest();
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

    auto [inputTestSignals, outputTestSignals] = createDataset(0., 100., 1000, INPUT_COUNT, rng, sqrt);

    for (size_t i = 0; i < inputTestSignals.size(); ++i) {
        inputLayer = inputTestSignals[i];
        h1 = inputLayer;
        forwardPropagation(hiddenLayers, h1, h2);

        for (size_t j = 0; j < outputLayer.size(); ++j) {
            outputLayer[j].setInputs(h2);
            outputLayer[j].updateSum();

            double estimator = outputTestSignals[i][j];
            double response = sigmoid::function(outputLayer[j].sum) * 10;
            double error = estimator - response;

            if (std::abs(error) < 0.5) {
                ++positiveAnswers;
            }
            else
            {
                ++negativeAnswers;
            }
        }
    }

    return std::make_pair(positiveAnswers, negativeAnswers);
}
