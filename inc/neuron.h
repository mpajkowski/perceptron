#pragma once

#include <vector>
#include <random>

struct Neuron
{
    Neuron(size_t inputCount, std::mt19937&);
    void setInputs(std::vector<double>&&);
    void update(double const& momentum, double const& learningRate, bool const& withBias);
    std::mt19937& rng;
    std::uniform_real_distribution<> dist;
    std::vector<double> inputs;
    std::vector<double> weights;
    std::vector<double> pWeights;
    double biasWeight;
    double biasPWeight;
    double error;
    double output;

private:
    void randomWeights();
};

