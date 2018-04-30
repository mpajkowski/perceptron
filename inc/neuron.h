#pragma once

#include <vector>
#include <random>

struct Neuron
{
    Neuron(size_t inputCount, std::mt19937&);
    std::mt19937& rng;
    std::normal_distribution<double> dist;
    //std::uniform_real_distribution<double> dist;
    std::vector<double> weights;
    std::vector<double> pWeights;
    double biasWeight;
    double biasPWeight;
    double error;
    double sum;
    double output;

    friend class Serializer;
};

