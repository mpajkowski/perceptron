#pragma once
#include <tinyxml2.h>
#include <fstream>
#include <memory>
#include "application.h"

class Serializer
{
public:
    Serializer(Application*);

    void saveData();
    void loadData();
private:
    tinyxml2::XMLDocument* xmlDoc;
    Application* application;
};
