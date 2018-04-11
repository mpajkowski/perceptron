#pragma once

#include <vector>
#include <random>

struct Neuron
{
    Neuron(size_t inputCount, bool isBias, std::mt19937&);
    void setInputs(std::vector<double> const&);
    void updateSum();
    void update();
    std::mt19937& rng;
    std::uniform_real_distribution<> dist;
    std::vector<double> inputs;
    std::vector<double> weights;
    double sum;
    double learnF;
    double delta;

private:
    void randomWeights();
};
