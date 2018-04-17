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
    for (size_t i = 0; i < layerConfiguration[0]; ++i) {
        inputLayer.emplace_back();
    }

    bool populateHidden = layerConfiguration.size() > 2;

    if (populateHidden) {
        for (size_t i = 1; i < layerConfiguration.size() - 1; ++i) {
            hiddenLayers.emplace_back();
            for (size_t j = 0; j < layerConfiguration[i]; ++j) {
                hiddenLayers[i - 1].emplace_back(i == 1 ? inputLayer.size()
                                                        : hiddenLayers[i - 2].size(),
                                                 rng);
            }
        }
    }

    for (size_t i = 0; i < layerConfiguration.back(); ++i) {
        outputLayer.emplace_back(populateHidden ? hiddenLayers.back().size()
                                                : inputLayer.size(),
                                 rng);
    }
}

void Net::forwardPropagation()
{
    for (size_t i = 0; i < hiddenLayers[0].size(); ++i) {
        hiddenLayers[0][i].setInputs(std::move(std::vector<double>(inputLayer)));

        double sum = .0;
        for (size_t j = 0; j < hiddenLayers[0][i].inputs.size(); ++j) {
            sum += hiddenLayers[0][i].weights[j] * hiddenLayers[0][i].inputs[j];
        }
        if (biasPresent) sum += hiddenLayers[0][i].biasWeight;
        hiddenLayers[0][i].output = sigmoid::function(sum);
    }

    // Layers input/output matcher
    auto perLayerInputMatch = [](layer_t const& previousLayer)
    {
        std::vector<double> previousLayerOutputs;
        for (size_t i = 0; i < previousLayer.size(); ++i) {
            previousLayerOutputs.push_back(previousLayer[i].output);
        }
        return previousLayerOutputs;
    };

    for (size_t i = 1; i < hiddenLayers.size(); ++i) {
        for (size_t j = 0; j < hiddenLayers[i].size(); ++j) {
            hiddenLayers[i][j].setInputs(
                    std::move(perLayerInputMatch(hiddenLayers[i - 1]))
            );

            double sum = .0;
            for (size_t k = 0; k < hiddenLayers[i][j].inputs.size(); ++k) {
               sum += hiddenLayers[i][j].inputs[k] * hiddenLayers[i][j].weights[k];
            }
            if (biasPresent) sum += hiddenLayers[i][j].biasWeight;
            hiddenLayers[i][j].output = sigmoid::function(sum);
        }
    }

    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i].setInputs(
                std::move(perLayerInputMatch(hiddenLayers.back()))
        );

        double sum = .0;
        for (size_t j = 0; j < outputLayer[i].inputs.size(); ++j) {
            sum += outputLayer[i].inputs[j] * outputLayer[i].weights[j];
        }
        if (biasPresent) sum += outputLayer[i].biasWeight;
        outputLayer[i].output = sigmoid::function(sum);
    }
}

double Net::calculateOutputError(std::vector<double> const& trainingSet)
{
    double globalError = .0;
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        double localError = .0;
        localError = trainingSet[i] - outputLayer[i].output;
        outputLayer[i].error = localError * sigmoid::derivative(outputLayer[i].output);
        globalError += localError * localError;
    }

    return globalError * .5;
}

void Net::backPropagation()
{
    size_t i = hiddenLayers.size() - 1;
    for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
        for (size_t k = 0; k < outputLayer.size(); k++) {
            hiddenLayers[i][j].error += outputLayer[k].error * outputLayer[k].weights[j];
        }
        hiddenLayers[i][j].error *= sigmoid::derivative(hiddenLayers[i][j].output);
    }

    while (i --> 0) {
        for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
            for (size_t k = 0; k < hiddenLayers[i + 1].size(); k++) {
                hiddenLayers[i][j].error +=
                    hiddenLayers[i + 1][k].error * hiddenLayers[i + 1][k].weights[j];
            }
            hiddenLayers[i][j].error *= sigmoid::derivative(hiddenLayers[i][j].output);
        }
    }
}

void Net::updateNeurons()
{
    for (auto& layer : hiddenLayers) {
        for (auto& neuron : layer) {
            neuron.update(momentum, learnF, biasPresent);
        }
    }

    for (auto& neuron : outputLayer) {
        neuron.update(momentum, learnF, biasPresent);
    }
}

double Net::run(std::vector<double> & input,
                std::vector<double> & output,
                bool train)
{
    inputLayer = input;
    forwardPropagation();
    double globalError = calculateOutputError(output);
    if (train) {
        backPropagation();
        updateNeurons();
    }
    return __builtin_expect(train, 1) ? calculateOutputError(output) : globalError;
}
