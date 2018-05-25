#pragma once

#include "func.h"
#include "fileLogger.h"
#include "net.h"

#include <memory>
#include <random>

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
    
    dataset_t inputLearn;
    dataset_t outputLearn;
    dataset_t inputTest;
    dataset_t outputTest;
    dataset_t inputValidate;
    dataset_t outputValidate;

    Logger* logger;
    std::string serializerPath;
    std::string loggerPath;
    friend class Serializer;
};
