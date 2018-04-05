#pragma once

#include <vector>
#include "neuron.h"

using layer_t = std::vector<Neuron>;

void forwardPropagation(std::vector<layer_t>& hiddenLayer, std::vector<double>& h1, std::vector<double>& h2);
void backpropOutput(layer_t& outputLayer, std::vector<double>& h2, std::vector<double> const& training_set);
void backpropHidden(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer);
void updateNeurons(std::vector<layer_t>& hiddenLayers, layer_t& outputLayer);

