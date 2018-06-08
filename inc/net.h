#pragma once

#include "neuron.h"

#include <random>
#include <vector>

class Logger;

using layer_t = std::vector<Neuron>;

class Net
{
public:
    Net(bool biasPresent, double momentum, double learnF,
        std::vector<size_t> const& layerConfiguration,
        std::mt19937& rng,
        Logger& logger);
    enum class Mode : std::uint32_t {
        LEARN,
        TEST,
        VALIDATE
    };
    double run(std::vector<double> const& input,
               std::vector<double> const& output,
               Mode mode);
private:
    void init(int argc, char* argv[]);
    void populateLayers(std::vector<size_t> const& layerConfiguration);
    void forwardPropagation();
    double calculateOutputError(std::vector<double> const& trainingSet, Mode mode);
    void backPropagation();
    void updateNeurons();

    bool biasPresent;
    double momentum;
    double learnF;
    std::vector<layer_t> layers;
    std::mt19937& rng;
    Logger& logger;

    friend class Serializer;
};
