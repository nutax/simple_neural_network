#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

/*
Begin license text.

Copyright 2022 JOSE IGNACIO HUBY OCHOA (GitHub: nutax)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

End license text.
*/



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
// Include: Standar Library
// -----------------------------
#include <functional>
#include <stdlib.h>



// -----------------------------
// Metaprogramming Utilities
// -----------------------------
template<unsigned... VALUES>
struct meta_arr;

template<typename X, typename Y>
struct weights_space_f;

template<unsigned... X, unsigned... Y>
struct weights_space_f<meta_arr<X...>, meta_arr<Y...>>{
static constexpr unsigned value = (((X+1)*Y) + ...);
};



// -----------------------------
// Neural Network Class
// -----------------------------
template<typename float_t, unsigned INPUTS, unsigned OUTPUTS, unsigned... HIDDENS>
class neural_network{

static constexpr unsigned LAYERS = sizeof...(HIDDENS) + 1;
static constexpr unsigned WEIGHTS_SPACE = weights_space_f<meta_arr<INPUTS, HIDDENS...>, meta_arr<HIDDENS..., OUTPUTS>>::value;
static constexpr unsigned NEURONS = (HIDDENS + ...) + OUTPUTS;

unsigned layer_size[LAYERS+1] = {INPUTS, HIDDENS..., OUTPUTS}; // layer + 1 -> layer size
unsigned nidx[LAYERS]; // layer -> neuron index
unsigned widx[LAYERS]; // layer -> weight index
float_t weight[WEIGHTS_SPACE]; //weight index -> weight
float_t output[NEURONS]; // neuron index -> neuron output
float_t error[NEURONS]; // neuron index -> neuron error
float_t delta[NEURONS]; // neuron index -> neuron delta

std::function<float_t(float_t)> const act;
std::function<float_t(float_t)> const dact;

public:
neural_network(std::function<float_t(float_t)> const & act, std::function<float_t(float_t)> const & dact): act{act}, dact{dact}{
	nidx[0] = 0;
	for(unsigned layer = 1; layer<LAYERS; ++layer){
		nidx[layer] = layer_size[layer] + nidx[layer-1];
	}
	widx[0] = 0;
	for(unsigned layer = 1; layer<LAYERS; ++layer){
		widx[layer] = (layer_size[layer-1]+1)*layer_size[layer] + widx[layer-1];
	}
	for(unsigned i = 0; i<WEIGHTS_SPACE; ++i){
		weight[i] = ((float_t)rand())/((float_t)RAND_MAX);
	}
}
float_t const * predict(float_t const input[]){
	for(int layer = 0; layer < LAYERS; ++layer){
		unsigned const n_inputs = layer_size[layer];
		unsigned const n_outputs = layer_size[layer+1];
		float_t const * const w = weight + widx[layer];
		
		float_t * const out = output + nidx[layer];

		for(unsigned neuron = 0; neuron < n_outputs; ++neuron){
			float_t activation = 0;
			float_t const * const nw = w + neuron*(n_inputs+1);
			unsigned i;
			for(i = 0; i<n_inputs; ++i){
				activation += input[i]*nw[i];
			}
			activation += 1*nw[i]; //bias
			out[neuron] = act(activation);
		}
		
		input = out;
	}
	return input;
}
void train(float_t const input[], float_t const *answer, float_t rate){
	auto _dummy = predict(input);
	// Tune output layer
	{
		unsigned const layer = LAYERS-1;
		unsigned const n_inputs = layer_size[layer];
		unsigned const n_outputs = layer_size[layer+1];
		float_t const * const in = output + nidx[layer-1];
		float_t const * const out = output + nidx[layer];
		
		float_t * const d = delta + nidx[layer];
		float_t * const w = weight + widx[layer];
		
		for(unsigned neuron = 0; neuron < n_outputs; ++neuron){
			d[neuron] = (answer[neuron]-out[neuron])*dact(out[neuron]);
			float_t * const nw = w + neuron*(n_inputs+1);
			unsigned i;
			for(i = 0; i<n_inputs; ++i){
				nw[i] += in[i] * d[neuron] * rate;
			}
			nw[i] += 1*d[neuron]*rate; //bias
		}
	}

	// Tune hidden layers
	for(int layer = LAYERS - 2; layer >= 0; --layer){
		unsigned const n_inputs = layer_size[layer];
		unsigned const n_outputs = layer_size[layer+1];
		unsigned const n_next_outputs = layer_size[layer+2];
		float_t const * const in = (layer) ? output + nidx[layer-1] : input;
		float_t const * const out = output + nidx[layer];
		float_t const * const next_d = delta + nidx[layer+1];
		float_t const * const next_w = weight + widx[layer+1];
		
		float_t * const d = delta + nidx[layer];
		float_t * const w = weight + widx[layer];

		for(int neuron = 0; neuron < n_outputs; ++neuron){
			float_t error = 0;
			for(int next_neuron = 0; next_neuron < n_next_outputs; ++next_neuron){
				float_t const * const next_nw = next_w + next_neuron*(n_outputs+1);
				error += next_d[next_neuron] * next_nw[neuron];
			}
			d[neuron] = error*dact(out[neuron]);
			float_t * const nw = w + neuron*(n_inputs+1);
			unsigned i;
			for(i = 0; i<n_inputs; ++i){
				nw[i] += in[i] * d[neuron] * rate;
			}
			nw[i] += (1)*(d[neuron])*rate; //bias
		}
		}
}

};

#endif