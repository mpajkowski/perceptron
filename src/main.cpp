#include "net.h"


int main(int argc, char* argv[])
{
    Net n{argc, argv};
    n.training(2200);
}
