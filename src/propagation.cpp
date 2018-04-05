#include "propagation.h"
#include <algorithm>
#include <boost/range/adaptor/reversed.hpp>

void forwardPropagation(std::vector<layer_t>& hiddenLayers, std::vector<double>& h1, std::vector<double>& h2)
{
    size_t i = 0;
    for (auto& layer : hiddenLayers) {
        for (auto& neuron : layer) {
            neuron.setInputs(h1);
            static_cast<void>(neuron.getSum());
            h2[i++] = neuron.activate();
        }
        h1 = h2;
    }
}

void backpropOutput(layer_t& outputLayer, std::vector<double>& h2, int i, std::vector<std::vector<double>> const& trainingSet)
{
    size_t j = 0;
    for (auto& neuron : outputLayer) {
        neuron.setInputs(h2);
        neuron.getSum();
        neuron.setGamma(((trainingSet[i][j++] / 10) - neuron.activate()) * neuron.derivative());
    }
}

void backpropHidden(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer)
{
    for (auto& hiddenLayer : boost::adaptors::reverse(hiddenLayers)) {
        for (auto& hiddenNeuron : hiddenLayer) {
            for (auto& outputNeuron : outputLayer) {
                // TODO backprop
            }
        }
    }
}

void updateNeurons(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer)
{
    for (auto& layer : hiddenLayers) {
        for (auto& neuron : layer) {
            neuron.update();
        }
    }

    for (auto& neuron : outputLayer) {
        neuron.update();
    }
}
