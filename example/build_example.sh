#!/bin/bash

# clean
mkdir -p build
cd build

cmake ..

# build
make clean
make -j4

cd ..

rm -rf build