#pragma once
#include <vector>
#include <random>
#include "neuron.h"

using layer_t = std::vector<Neuron>;

class Net
{
public:
    Net(int argc, char* argv[]);
    void training();

    std::pair<int, int> test();
private:
    void init(int argc, char* argv[]);
    void populateLayers(std::vector<size_t> const& layerConfiguration);
    void forwardPropagation();
    void backPropagation(std::vector<double> const& trainingSet);
    void updateNeurons();

    size_t trainingEpochs;
    bool biasPresent;
    double momentum;
    double learnF;
    std::vector<double> inputLayer;
    std::vector<layer_t> hiddenLayers;
    layer_t outputLayer;
    std::vector<double> prod;
    std::mt19937 rng;
};

