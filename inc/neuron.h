#pragma once

#include <vector>
#include <random>

class Neuron
{
public:
    Neuron(size_t inputCount, bool isBias, std::mt19937&);
    void setInputs(std::vector<double> const&);
    void setDelta(double val);
    void increaseLearnFactor();
    void updateSum();
    void update();
    double getWeight(size_t i);
    double& getSum();
    double getDelta();
private:
    std::mt19937& rng;
    std::uniform_real_distribution<> dist;
    std::vector<double> inputs;
    std::vector<double> weights;
    double sum;
    double learnF;
public:
    double delta;
private:
    void randomWeights();
};
