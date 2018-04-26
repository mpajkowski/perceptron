#include "fileLogger.h"

FileLogger::FileLogger(std::string const& path)
    : stream{new std::ofstream}
{
    if (path != "") {
        stream->open(path);
    }
}

FileLogger::~FileLogger()
{
    stream->close();
    delete stream;
}

void FileLogger::addToStream(std::string const& str)
{
    *stream << str << std::endl;
}

