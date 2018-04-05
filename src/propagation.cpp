#include "propagation.h"
#include <algorithm>
#include <boost/range/adaptor/reversed.hpp>
#include <cassert>

void forwardPropagation(std::vector<layer_t>& hiddenLayers, std::vector<double>& h1, std::vector<double>& h2)
{
    assert(h1.size() > 0);

    for (auto& layer : hiddenLayers) {
        for (size_t i = 0; i < layer.size(); ++i) {
            layer[i].setInputs(h1);
            layer[i].getSum();
            h2[i] = layer[i].activate();
        }
        h1 = h2;
    }
}

void backpropOutput(layer_t& outputLayer, std::vector<double>& h2, std::vector<double> const& trainingSet)
{
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i].setInputs(h2);
        outputLayer[i].getSum();
        outputLayer[i].setGamma(((trainingSet[i] / 10) - outputLayer[i].activate()) * outputLayer[i].derivative());
    }
}

void backpropHidden(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer)
{
    for (auto& hiddenLayer : boost::adaptors::reverse(hiddenLayers)) {
        for (size_t i = 0; i < hiddenLayer.size(); ++i) {
            for (size_t j = 0; j < outputLayer.size(); ++j) {
                auto tmp = hiddenLayer[i].getGamma();
                hiddenLayer[i].setGamma(tmp + hiddenLayer[i].getGamma()
                                        * outputLayer[j].getWeight(i));
            }
            auto tmp = hiddenLayer[i].getGamma();
            hiddenLayer[i].setGamma(tmp * hiddenLayer[i].derivative());
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
