#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, bool isBias, double learnF,
               double momentum, std::mt19937& rng)
    : sum{.0}
    , delta{.0}
    , rng{rng}
    , dist{-.5, .5}
    , learnF{learnF}
    , momentum{momentum}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    inputs.resize(inputCount);
    prevWeights.resize(inputs.size());
    std::fill(prevWeights.begin(), prevWeights.end(), .0);
}

void Neuron::setInputs(std::vector<double> const& src)
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = src[i];
    }
}

void Neuron::updateSum()
{
    sum = .0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i] * weights[i];
    }
}

void Neuron::update()
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] = weights[i] + delta * learnF * inputs[i] + momentum * prevWeights[i];
        prevWeights[i] = learnF * delta * inputs[i];
    }
    delta = .0;
}
