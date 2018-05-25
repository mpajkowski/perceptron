#include "application.h"
#include "serializer.h"

#include <iostream>
#include <memory>

int main(int argc, char* argv[])
{
    auto app = std::make_unique<Application>(argc, argv);
    app->runNetwork(true);
    app->runNetwork(false);
    auto serializer = std::make_unique<Serializer>(app.get());
    serializer->saveData();
}

