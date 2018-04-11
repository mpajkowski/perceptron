#include "net.h"
#include "csv.h"

int main(int argc, char* argv[])
{
    Net n{argc, argv};
    n.training();
}
