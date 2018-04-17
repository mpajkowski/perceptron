#include <iostream>
#include <memory>
#include "application.h"

int main(int argc, char* argv[])
{
    auto app = std::make_unique<Application>(argc, argv);
    app->runNetwork(true);
    app->runNetwork(false);
}

