#pragma once
#include <vector>
#include <random>
#include "neuron.h"

using layer_t = std::vector<Neuron>;
class Logger;

class Net
{
public:
    Net(bool biasPresent, double momentum, double learnF,
        std::vector<size_t> const& layerConfiguration,
        std::mt19937& rng,
        Logger& logger);
    double run(std::vector<double> const& input,
               std::vector<double> const& output,
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
    double globalError;
    std::vector<layer_t> layers;
    std::mt19937& rng;
    Logger& logger;

    friend class Serializer;
};

