#include "net.h"
#include "func.h"
#include "neuron.h"
#include "fileLogger.h"

#include <boost/program_options.hpp>
#include <cstdlib>

Net::Net(bool biasPresent, double momentum, double learnF,
         std::vector<size_t> const& layerConfiguration,
         std::mt19937& rng,
         Logger& logger)
    : biasPresent{biasPresent}
    , momentum{momentum}
    , learnF{learnF}
    , rng{rng}
    , logger{logger}
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
        auto& currLayer = layers[i];
        auto& prevLayer = layers[i - 1];

        for (auto& currLayerNeuron : currLayer) {
            double sum = .0;

            for (size_t k = 0; k < prevLayer.size(); ++k) {
               sum += prevLayer[k].output * currLayerNeuron.weights[k];
            }

            if (biasPresent) {
                sum += currLayerNeuron.biasWeight;
            }

            currLayerNeuron.output = sigmoid::function(sum);
        }
    }
}

double Net::calculateOutputError(std::vector<double> const& trainingSet,
                                 Mode mode)
{
    double globalError = .0;
    auto& outputLayer = layers.back();
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        double const& output = outputLayer[i].output;
        double localError = trainingSet[i] - output;

        if (logger.isVerbose() || mode == Mode::VALIDATE) {
            logger.addToStream({"Expected: " + std::to_string(trainingSet[i])
                + ", output: " + std::to_string(output)});
        }

        outputLayer[i].error = output * (1. - output) * localError;
                 // derivative ^^^^^^^^^^^^^^^^^^^^^^
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
                   // derivative ^^^^^^^^^^^^^^^^^^^^^^
        }
    }
}

void Net::updateNeurons()
{
    for (size_t i = 1; i < layers.size(); ++i) {
        auto& currLayer = layers[i];
        auto& prevLayer = layers[i - 1];

        for (size_t j = 0; j < currLayer.size(); ++j) {
            for (size_t k = 0; k < prevLayer.size(); ++k) {
                currLayer[j].weights[k] += learnF
                                         * currLayer[j].error
                                         * prevLayer[k].output 
                                         + momentum
                                         * currLayer[j].pWeights[k];
                currLayer[j].pWeights[k] = learnF
                                         * currLayer[j].error
                                         * prevLayer[k].output;
            }

            if (biasPresent) {
                currLayer[j].biasWeight += currLayer[j].error * learnF
                                         + momentum * currLayer[j].biasPWeight;
                currLayer[j].biasPWeight = learnF * currLayer[j].error;
            }
        }
    }
}

double Net::run(std::vector<double> const& input,
                std::vector<double> const& output,
                Mode mode)
{
    std::ostringstream attributes;

    if (logger.isVerbose() || mode == Mode::VALIDATE) {
        attributes << "=== Attributes: ";
    }

    for (size_t i = 0; i < input.size(); ++i) {
        layers[0][i].output = input[i];
        if (logger.isVerbose() || mode == Mode::VALIDATE) {
            attributes << input[i] << " ";
        }
    }

    if (logger.isVerbose() || mode == Mode::VALIDATE) {
        logger.addToStream(attributes.str());
    }

    forwardPropagation();

    double globalError = calculateOutputError(output, mode);

    if (mode == Mode::LEARN) {
        backPropagation();
        updateNeurons();
    }

    return globalError;
}
