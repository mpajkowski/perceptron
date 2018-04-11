#include "propagation.h"
#include "activationFunction.h"

#include <algorithm>
#include <cassert>

void forwardPropagation(std::vector<layer_t>& hiddenLayers, std::vector<double>& h1, std::vector<double>& h2)
{
    assert(h1.size() > 0);

    for (auto& layer : hiddenLayers) {
        for (size_t i = 0; i < layer.size(); ++i) {
            layer[i].setInputs(h1);
            layer[i].updateSum();
            h2[i] = sigmoid::function(layer[i].sum);
        }
        h1 = h2;
    }
}

void backpropOutput(layer_t& outputLayer, std::vector<double>& h2, std::vector<double> const& trainingSet)
{
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        outputLayer[i].setInputs(h2);
        outputLayer[i].updateSum();
        outputLayer[i].delta = ((trainingSet[i] / 10) - sigmoid::function(outputLayer[i].sum))
          * sigmoid::derivative(outputLayer[i].sum);
    }
}

void backpropHidden(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer)
{
    size_t i = hiddenLayers.size() - 1;
    for (size_t j = 0; j < hiddenLayers[i].size(); j++) {
        for (size_t k = 0; k < outputLayer.size(); k++) {
            hiddenLayers[i][j].delta += outputLayer[k].delta * outputLayer[k].weights[j];
        }
        hiddenLayers[i][j].delta *= sigmoid::derivative(hiddenLayers[i][j].sum);
    }

    while (i --> 0) {
        for (size_t j = 0;j<hiddenLayers[i].size();j++) {
            for (size_t k = 0; k < hiddenLayers[i + 1].size(); k++) {
                hiddenLayers[i][j].delta += hiddenLayers[i + 1][k].delta * hiddenLayers[i + 1][k].weights[j];
            }
            hiddenLayers[i][j].delta *= sigmoid::derivative(hiddenLayers[i][j].sum);
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
