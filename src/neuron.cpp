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
    , sum{.0}
{
    for (size_t i = 0; i < inputCount; ++i) {
        weights.push_back(dist(rng));
    }

    pWeights.resize(inputCount);
    std::fill(std::begin(pWeights), std::end(pWeights), .0);
    biasWeight = dist(rng);
}

