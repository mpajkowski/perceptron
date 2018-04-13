#include "net.h"
#include "csv.h"
#include <array>
#include <iostream>

int main(int argc, char* argv[])
{
    Net n{argc, argv};
    n.training();
    
    io::CSVReader<7> in{"../data/iris.csv"};
    double data[150][7];
    for (size_t i = 0; in.read_row(data[i][0], data[i][1], data[i][2],
                       data[i][3], data[i][4], data[i][5],
                       data[i][6]); ++i);

 //   for (size_t i = 0; i < 150; ++i) {
 //       for (size_t j = 0; j < 7; ++j) {
 //           std::cout << data[i][j] << (j == 6 ? "" : ",");
 //       }
 //       std::cout << std::endl;
 //   }
}
