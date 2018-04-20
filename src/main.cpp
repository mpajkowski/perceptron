#include <iostream>
#include <memory>
#include "application.h"
#include "serializer.h"

int main(int argc, char* argv[])
{
    auto* app = new Application{argc, argv};
    app->runNetwork(true);
    auto serializer = std::make_unique<Serializer>(app);
    serializer->saveData();
    delete app;
}

