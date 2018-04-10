#include "activationFunction.h"
#include <cmath>

// Dummy function for memtests
// double std::exp(double) does not like valgrind
double dummy(double x) { return 0; }

double sigmoid::function(double x)
{
    return 1. / (1. + std::exp(-x));
}

double sigmoid::derivative(double x)
{
    return sigmoid::function(x) * (1 - sigmoid::function(x));
}

double relu::function(double x)
{
    return 1 - (x < 0 ? .01 * x : x);
}

double relu::derivative(double x)
{
    return 1 - (x < 0 ? .01 : 1);
}
