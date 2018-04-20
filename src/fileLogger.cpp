#include "fileLogger.h"

FileLogger::FileLogger()
{
    this->stream.open("report.txt");
}

FileLogger::~FileLogger()
{
    stream.close();
}

void FileLogger::addToStream(std::string const& str)
{
    stream << str;
}

