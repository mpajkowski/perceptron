#pragma once
#include <memory>
#include <random>
#include "net.h"
#include "fileLogger.h"

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
    size_t probingFreq;
    bool verboseOutput;
    std::mt19937 rng;
    Logger* logger;
    std::string serializerPath;
    std::string loggerPath;
    friend class Serializer;
};

