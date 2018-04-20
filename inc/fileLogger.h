#pragma once
#include <fstream>

class FileLogger
{
public:
    FileLogger(std::string const& path);
    ~FileLogger();
    void addToStream(std::string const&);
private:
    std::ofstream* stream;
};
