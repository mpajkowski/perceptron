#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, bool isBias, std::mt19937& rng)
    : sum{.0}
    , delta{.0}
    , rng{rng}
    , dist{-1., 1.}
    , learnF{.01}
    , isBias{isBias}
    , inputCount{inputCount}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    inputs.resize(inputCount);
}

void Neuron::setInputs(std::vector<double> const& src)
{
    inputs = src;
}

void Neuron::setDelta(double val)
{
    delta = val;
}

void Neuron::increaseLearnFactor()
{
    learnF += .0004;
}

double Neuron::getSum()
{
    return sum;
}

void Neuron::updateSum()
{
    sum = .0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }
}

double Neuron::getDelta()
{
    return delta;
}

double Neuron::getWeight(size_t i)
{
    return weights[i];
}

void Neuron::update()
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] += delta * learnF * inputs[i];
    }
}
