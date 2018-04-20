#pragma once
#include <fstream>

class FileLogger
{
public:
    FileLogger();
    ~FileLogger();
    void addToStream(std::string const&);
private:
    std::ofstream* stream;
};
