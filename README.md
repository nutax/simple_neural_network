# Simple NN: Multilayer Perceptron
Simple neural network MLP with modern C++, metaprogramming and vector optimizations.

## Compilation flags
-std=c++17 -Ofast -march=native

## Who this is for?
Students.

## Why metaprogramming?
To statically define the memory needed and to unroll loops to help the compiler vectorize the code.

## External SIMD (loop unrolling) library
- [Pure SIMD](https://github.com/eatingtomatoes/pure_simd) by [eatingtomatoes](https://github.com/eatingtomatoes): Simple C++17 metaprogramming tools to unroll loops

## Warnings
1. I'm still learning the basics of NN.
2. Haven't tested multilayers enough.

## Recommended reading
- [Simple neural network implementation in C](https://medium.com/towards-data-science/simple-neural-network-implementation-in-c-663f51447547) by Santiago Becerra
