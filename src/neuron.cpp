#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, std::mt19937& rng)
    : output{.0}
    , error{.0}
    , rng{rng}
    , dist{-.5, .5}
    , biasPWeight{.0}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    inputs.resize(inputCount);

    pWeights.resize(inputs.size());
    std::fill(pWeights.begin(), pWeights.end(), .0);
    biasWeight = dist(rng);
}

void Neuron::setInputs(std::vector<double>&& src)
{
    inputs = src;
}

void Neuron::update(double const& momentum, double const& learningRate,
                    bool const& withBias)
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        weights[i] += error * learningRate * inputs[i]
                   + momentum * pWeights[i];
        pWeights[i] = learningRate * error * inputs[i];

    }
    
    if (withBias) {
        biasWeight += error * learningRate + momentum * biasPWeight;
        biasPWeight = learningRate * error;
    }
}
