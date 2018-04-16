#include "net.h"
#include "csv.h"
#include <array>
#include <iostream>

int main(int argc, char* argv[])
{
    Net {argc, argv}
        .training();
}

