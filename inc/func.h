#pragma once

#include <cassert>
#include <vector>
#include <functional>
#include "propagation.h"

constexpr size_t INPUT_COUNT = 1;

using dataset_t = std::vector<std::vector<double>>;

std::pair<dataset_t, dataset_t>
createDataset(double rangeMin, double rangeMax,
              size_t setSize, size_t inputCount,
              std::mt19937& rng,
              std::function<double(double)> callback);

void adjustHelpers(std::vector<double>& inputLayer,
                   std::vector<layer_t>& hiddenLayers,
                   layer_t& outputLayer,
                   std::vector<double>& h1,
                   std::vector<double>& h2);

void training(size_t epochs, std::vector<double>& inputLayer,
              std::vector<layer_t>& hiddenLayers,
              layer_t& outputLayer,
              std::vector<double>& h1,
              std::vector<double>& h2,
              std::mt19937& rng);

std::pair<int, int> test(std::vector<double>& inputLayer,
                         std::vector<layer_t>& hiddenLayers,
                         layer_t& outputLayer,
                         std::vector<double>& h1,
                         std::vector<double>& h2,
                         std::mt19937& rng);
