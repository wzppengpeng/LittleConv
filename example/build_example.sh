#!/bin/bash

# clean
mkdir -p build
cd build

cmake ..

# build
make -j4

cd ..
