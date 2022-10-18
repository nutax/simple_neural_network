/*
	Created by José Ignacio Huby Ochoa (GitHub: nutax).
	Last update: 18/10/2022.

	Lang: C++17.

	Warning #1: I'm still learning the basics.
	Warning #2: Haven't tested multilayers enough.

	TODO:
	- More tests with multiple hidden layers
	- Make it more customizable
	- Support CPU SIMD
	- Support GPU
*/



// -----------------------------
// Include: Standard Library
// -----------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



// -----------------------------
// Include: User Library (NN)
// -----------------------------
#include "neural_network.hpp"



// -----------------------------
// Procedures Declaration
// -----------------------------
int main();
void shuffle(int *array, int n);
float sigmoid(float x);
float dsigmoid(float x);



// -----------------------------
// Training Configuration
// -----------------------------
int const rand_seed = 0;
int const epochs = 10000;
int const rows = 4;
int const cols = 2;
float const learning_rate = 0.1;

float const train_input[] = {1, 1, 1, 0, 0, 1, 0, 0};
float const train_output[] = {0, 1, 1, 0}; // XOR problem

int order[] = {0, 1, 2, 3};



// -----------------------------
// AI Model: Neural Network
// -----------------------------
neural_network<float, 2, 1, 2> nn(sigmoid, dsigmoid);
/*
	Where: < float_t, INPUTS, OUTPUTS, HIDDENS...> nn(activation_f, derivate_activation_f);
	Example: < double, 2, 1, 3, 2> == two inputs neurons, one output neuron, three neurons in first hidden layer, two neurons in second hidden layer
	Warning: Haven't tested multilayers enough.
*/



// -----------------------------
// Procedures Definition
// -----------------------------
int main() {
  srand(rand_seed);
  for (int i = 0; i < epochs; i++) {
    shuffle(order, rows);
    for (int j = 0; j < rows; j++) {
      int row = order[j];
      nn.train(train_input + row * cols, train_output + row, learning_rate);
    }
  }
  for (int j = 0; j < rows; ++j) {
    int row = order[j];
    float_t const *pred = nn.predict(train_input + row * cols);
    printf("%f XOR %f == %f\n", train_input[row * cols],
           train_input[row * cols + 1], pred[0]);
  }
}

void shuffle(int *array, int n) {
  if (n > 1) {
    int i;
    for (i = 0; i < n - 1; i++) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float dsigmoid(float x) { return x * (1 - x); }