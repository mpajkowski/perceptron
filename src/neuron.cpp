#include "neuron.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

Neuron::Neuron(size_t inputCount, bool isBiasNeuron, double learnF,
               double momentum, std::mt19937& rng)
    : sum{.0}
    , error{.0}
    , rng{rng}
    , dist{-.5, .5}
    , learnF{learnF}
    , momentum{momentum}
    , isBiasNeuron{isBiasNeuron}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    inputs.resize(inputCount);

    pWeights.resize(inputs.size());
    std::fill(pWeights.begin(), pWeights.end(), .0);
}

void Neuron::setInputs(std::vector<double> const& src)
{
    if (!isBiasNeuron) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = src[i];
        }
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
        weights[i] = weights[i] + error * learnF * inputs[i] + momentum * pWeights[i];
        pWeights[i] = learnF * error * inputs[i];
    }
  //  error = .0;
}
