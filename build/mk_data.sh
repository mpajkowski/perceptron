#/bin/bash

for i in `ls ../data/e*`; do rm $i ; done

./perceptron -c 4 2 4 -e 5000 -l 0.9 -m 0.0             --log-learning ../data/e5000l09m00  --serialize ../data/e5000l09m00.xml
./perceptron -c 4 2 4 -e 5000 -l 0.6 -m 0.0             --log-learning ../data/e5000l06m00  --serialize ../data/e5000l06m00.xml
./perceptron -c 4 2 4 -e 5000 -l 0.2 -m 0.0             --log-learning ../data/e5000l02m00  --serialize ../data/e5000l02m00.xml
./perceptron -c 4 2 4 -e 5000 -l 0.9 -m 0.6             --log-learning ../data/e5000l09m06  --serialize ../data/e5000l09m06.xml
./perceptron -c 4 2 4 -e 5000 -l 0.2 -m 0.9             --log-learning ../data/e5000l02m09  --serialize ../data/e5000l02m09.xml
./perceptron -c 4 2 4 -e 5000 -l 0.9 -m 0.0 --with-bias --log-learning ../data/e5000l09m00b --serialize ../data/e5000l09m00b.xml
./perceptron -c 4 2 4 -e 5000 -l 0.6 -m 0.0 --with-bias --log-learning ../data/e5000l06m00b --serialize ../data/e5000l06m00b.xml
./perceptron -c 4 2 4 -e 5000 -l 0.2 -m 0.0 --with-bias --log-learning ../data/e5000l02m00b --serialize ../data/e5000l02m00b.xml
./perceptron -c 4 2 4 -e 5000 -l 0.9 -m 0.6 --with-bias --log-learning ../data/e5000l09m06b --serialize ../data/e5000l09m06b.xml
./perceptron -c 4 2 4 -e 5000 -l 0.2 -m 0.9 --with-bias --log-learning ../data/e5000l02m09b --serialize ../data/e5000l02m09b.xml
