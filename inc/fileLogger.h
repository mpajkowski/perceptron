#pragma once
#include <fstream>

class Logger
{
public:
    Logger(std::string const& path);
    ~Logger();
    void addToStream(std::string const&);
private:
    std::ostream* stream;
};
