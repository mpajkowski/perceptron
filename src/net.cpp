#include "net.h"
#include "func.h"
#include "neuron.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <cstdlib>

Net::Net(bool biasPresent, double momentum, double learnF,
         std::vector<size_t> const& layerConfiguration,
         std::mt19937& rng)
    : biasPresent{biasPresent}
    , momentum{momentum}
    , learnF{learnF}
    , rng{rng}
{
    populateLayers(layerConfiguration);
}

void Net::populateLayers(std::vector<size_t> const& layerConfiguration)
{
    for (size_t i = 0; i < layerConfiguration.size(); ++i) {
        layers.emplace_back();
        for (size_t j = 0; j < layerConfiguration[i]; ++j) {
            layers[i].emplace_back(i == 0 ? 0
                                          : layerConfiguration[i - 1],
                                   rng);
        }
    }
}

void Net::forwardPropagation()
{
    for (size_t i = 1; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].size(); ++j) {
            double sum = .0;
            for (size_t k = 0; k < layers[i - 1].size(); ++k) {
               sum += layers[i - 1][k].output * layers[i][j].weights[k];
            }
            if (biasPresent) sum += layers[i][j].biasWeight;
            layers[i][j].output = sigmoid::function(sum);
        }
    }
}

double Net::calculateOutputError(std::vector<double> const& trainingSet)
{
    double globalError = .0;
    auto& outputLayer = layers.back();
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        double const& output = outputLayer[i].output;
        double localError = trainingSet[i] - output;
        outputLayer[i].error = output * (1. - output) * localError;
        globalError += localError * localError;
    }

    return globalError * .5;
}

void Net::backPropagation()
{
    for (size_t i = layers.size() - 1; i > 1; --i) {
        auto& currLayer = layers[i - 1];
        auto& nextLayer = layers[i];

        for (size_t j = 0; j < currLayer.size(); ++j) {
            double const& output = currLayer[j].output;
            double err = .0;

            for (size_t k = 0; k < nextLayer.size(); ++k) {
                err += nextLayer[k].weights[j] * nextLayer[k].error;
            }

            currLayer[j].error = output * (1. - output) * err;
        }
    }
}

void Net::updateNeurons()
{
    for (size_t i = 1; i < layers.size(); ++i) {
        for (size_t j = 0; j < layers[i].size(); ++j) {
            for (size_t k = 0; k < layers[i - 1].size(); ++k) {
                layers[i][j].weights[k] += learnF
                                         * layers[i][j].error
                                         * layers[i - 1][k].output
                                         + momentum
                                         * layers[i][j].pWeights[k];
                layers[i][j].pWeights[k] = learnF
                                         * layers[i][j].error
                                         * layers[i - 1][k].output;
            }
            if (biasPresent) {
                layers[i][j].biasWeight += layers[i][j].error * learnF
                                         + momentum * layers[i][j].biasPWeight;
                layers[i][j].biasPWeight = learnF * layers[i][j].error;
            }
        }
    }
}

double Net::run(std::vector<double> const& input,
                std::vector<double> const& output,
                bool train)
{
    for (size_t i = 0; i < input.size(); ++i) {
        layers[0][i].output = input[i];
    }

    forwardPropagation();

    double globalError = calculateOutputError(output);

    if (train) {
        backPropagation();
        updateNeurons();
    }

    return globalError;
}
