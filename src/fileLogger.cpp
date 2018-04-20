#include "fileLogger.h"

FileLogger::FileLogger()
    : stream{new std::ofstream}
{
    stream->open("report.txt");
}

FileLogger::~FileLogger()
{
    stream->close();
    delete stream;
}

void FileLogger::addToStream(std::string const& str)
{
    *stream << str;
}

