#pragma once

#include <cassert>
#include <vector>
#include "propagation.h"

constexpr size_t NEURON_COUNT = 10;
constexpr size_t INPUT_COUNT = 4;

void training(size_t epochs, std::vector<double>& inputLayer,
              std::vector<layer_t>& hiddenLayers,
              layer_t& outputLayer,
              std::vector<std::vector<double>>& inputLearnSignals,
              std::vector<std::vector<double>>& outputLearnSignals,
              std::vector<double>& h1,
              std::vector<double>& h2);

std::pair<int, int> test(std::vector<double>& inputLayer,
                         std::vector<layer_t>& hiddenLayers,
                         layer_t& outputLayer,
                         std::vector<std::vector<double>>& inputTestSignals,
                         std::vector<std::vector<double>>& outputTestSignals,
                         std::vector<double>& h1,
                         std::vector<double>& h2);
