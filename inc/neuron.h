#pragma once

#include <vector>
#include <random>

class Neuron
{
public:
    Neuron(size_t inputCount, bool isBias, std::mt19937&);
    void setInputs(std::vector<double> const&);
    void setGamma(double val);
    double activate();
    void increaseLearnFactor();
    double derivative();
    void update();
    double getWeight(size_t i);
    double getSum();
    double getGamma();
private:
    std::mt19937& rng;
    std::uniform_real_distribution<> dist;
    std::vector<double> inputs;
    std::vector<double> weights;
    double sum;
    double learnF;
public:
    double gamma;
private:
    bool isBias;
    size_t inputCount;
    void randomWeights();
};
