#include "fileLogger.h"
#include <iostream>

Logger::Logger(std::string const& path, bool verbose, size_t probingFreq)
    : stream{nullptr}
    , verbose{verbose}
    , probingFreq{probingFreq}
    , itCounter{nullptr}
{
    if (path != "") {
        stream = new std::ofstream;
        dynamic_cast<std::ofstream*>(stream)->open(path);
    } else {
        stream = &std::cout;
    }
}

Logger::~Logger()
{
    if (auto st = dynamic_cast<std::ofstream*>(stream); st != nullptr) {
        st->close();
        delete stream;
    }
}

void Logger::addToStream(std::string const& str)
{
    if ((itCounter ? *itCounter : 0 ) % probingFreq == 0) {
      *stream << str << std::endl;
    }
}

void Logger::setItCounter(size_t* i)
{
    itCounter = i;
}

bool Logger::isVerbose() const
{
  return verbose;
}
