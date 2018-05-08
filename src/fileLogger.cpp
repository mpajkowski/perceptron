#include "fileLogger.h"
#include <iostream>

Logger::Logger(std::string const& path)
    : stream{nullptr}
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
    *stream << str << std::endl;
}

