#pragma once
#include <vector>
#include <random>
#include "neuron.h"

using layer_t = std::vector<Neuron>;

class Net
{
public:
    Net(bool biasPresent, double momentum, double learnF,
        std::vector<size_t> const& layerConfiguration,
        std::mt19937& rng);
    double run(std::vector<double> & input,
               std::vector<double> & output,
               bool train);
private:
    void init(int argc, char* argv[]);
    void populateLayers(std::vector<size_t> const& layerConfiguration);
    void forwardPropagation();
    double calculateOutputError(std::vector<double> const& trainingSet);
    void backPropagation();
    void updateNeurons();

    bool biasPresent;
    double momentum;
    double learnF;
    std::vector<double> inputLayer;
    std::vector<layer_t> hiddenLayers;
    layer_t outputLayer;
    std::mt19937& rng;
};

