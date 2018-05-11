#pragma once
#include <fstream>

class Logger
{
public:
    Logger(std::string const& path, bool verbose, size_t probingFreq);
    ~Logger();
    void addToStream(std::string const&);
    bool isVerbose() const;
    void setItCounter(size_t* i);
private:
    std::ostream* stream;
    bool verbose;
    size_t probingFreq;
    size_t* itCounter;
};
