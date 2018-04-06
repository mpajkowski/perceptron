#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, bool isBias, std::mt19937& rng)
    : sum{.0}
    , gamma{.0}
    , rng{rng}
    , dist{-1., 1.}
    , learnF{.01}
    , isBias{isBias}
    , inputCount{inputCount}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    assert(weights.size() == inputCount);
    inputs.resize(inputCount);
}

void Neuron::setInputs(std::vector<double> const& src)
{
    for (size_t i = 0; i < inputCount; ++i) {
        inputs[i] = src[i];
    }
}

void Neuron::setGamma(double val)
{
    gamma = val;
}

double Neuron::activate()
{
    return 1. / (1 + exp(-sum));
}

double Neuron::derivative()
{
    return activate() * (1 - activate());
}

void Neuron::increaseLearnFactor()
{
    learnF += .001;
}

double Neuron::getSum()
{
    sum = .0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
        if (isBias) {
           // sum += inputs[i]; // bias
        }
    }

    return sum;
}

double Neuron::getGamma()
{
    return gamma;
}

double Neuron::getWeight(size_t i)
{
    return weights[i];
}

void Neuron::update()
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] += gamma * learnF * inputs[i];
    }
}
