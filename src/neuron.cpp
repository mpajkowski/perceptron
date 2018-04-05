#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, bool isBias, std::mt19937& rng)
    : id{idCounter++}
    , sum{0.0}
    , gamma{1.0}
    , rng{rng}
    , dist{-1.0, 1.0}
    , learnF{0.001}
    , isBias{isBias}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    assert(weights.size() == inputCount);
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
    learnF += 0.0001;
}

double Neuron::getSum()
{
    sum = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
        if (isBias) {
       //     sum += inputs[i] / 2; // bias 0.5
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
        weights[i] = weights[i] + (gamma * learnF * inputs[i]);
    }
}

size_t Neuron::idCounter = 0;
