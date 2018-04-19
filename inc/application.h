#pragma once
#include <memory>
#include <random>
#include "net.h"

class Application
{
public:
    Application(int argc, char* argv[]);
    ~Application();
    void runNetwork(bool learn);
private:
    void init(int argc, char* argv[]);
    Net* net;
    size_t trainingEpochs;
    bool verboseOutput;
    std::mt19937 rng;
    friend class Serializer;
};

