#pragma once
#include <vector>
#include <random>
#include "neuron.h"

using layer_t = std::vector<Neuron>;

class Net
{
public:
    Net(int argc, char* argv[]);
    void training(size_t epochs);

    std::pair<int, int> test();
private:
    void init(int argc, char* argv[]);
    void forwardPropagation();
    void backPropagation(std::vector<double> const& trainingSet);
    void updateNeurons();
    void adjustHelpers();

    std::vector<double> inputLayer;
    std::vector<layer_t> hiddenLayers;
    layer_t outputLayer;
    std::vector<double> _1;
    std::vector<double> _2;
    std::mt19937 rng;
};