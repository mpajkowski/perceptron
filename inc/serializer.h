#pragma once
#include <tinyxml2.h>
#include <memory>
#include "application.h"

class Serializer
{
public:
    Serializer(Application*);
    void saveData();
    void readData();
private:
    tinyxml2::XMLDocument* xmlDoc;
    Application* application;
};
