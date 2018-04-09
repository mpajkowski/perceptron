#include "activationFunction.h"
#include <cmath>

double sigmoid::function(double x)
{
    return 1. / (1. + exp(-x));
}

double sigmoid::derivative(double x)
{
    return sigmoid::function(x) * (1 - sigmoid::function(x));
}

double relu::function(double x)
{
    return (x < 0 ? .01 * x : x);
}

double relu::derivative(double x)
{
    return (x < 0 ? .01 : 1);
}
