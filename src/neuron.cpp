#include "neuron.h"
#include <cmath>
#include <algorithm>

Neuron::Neuron(size_t inputCount, bool isBias, std::mt19937& rng)
    : id{idCounter++}
    , sum{0.0}
    , gamma{0.0}
    , rng{rng}
    , dist{-1.0, 1.0}
    , learnF{0.001}
{
    if (isBias) {
        ++inputCount;
    }

    weights.resize(inputCount);
    setWeights();
}

void Neuron::setWeights()
{
    for (auto& it : weights) {
        it = dist(rng);
    }
}

void Neuron::setInputs(std::vector<double> const& src)
{
    inputs = src;    
}

void Neuron::setGamma(double val)
{
    gamma = val;
}

double Neuron::activate()
{
    return 1.0 / (1 + exp(-sum));
}

double Neuron::derivative()
{
    return activate() * (1 - activate());
}

void Neuron::increaseLearnFactor()
{
    learnF += 10.0;
}

double Neuron::getSum()
{
    sum = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }

    return sum;
}

double Neuron::getGamma()
{
    return gamma;
}

void Neuron::update()
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] = weights[i] + (gamma * learnF * inputs[i]);
    }
}

size_t Neuron::idCounter = 0;
