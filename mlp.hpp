#ifndef __MLP_H__
#define __MLP_H__

#include <tuple>
#include <stdlib.h>
#include <math.h>
#include "pure_simd.hpp"

/*
MIT License

Copyright (c) 2022 Jose Ignacio Huby Ochoa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

namespace meta_ai
{
#define FAST_RAND_MAX 32767
    static unsigned int g_seed = 5;

    inline int fast_rand(void)
    {
        g_seed = (214013 * g_seed + 2531011);
        return (g_seed >> 16) & 0x7FFF;
    }

    namespace simd = pure_simd;

    template <std::size_t... Is>
    constexpr auto indexSequenceReverse(std::index_sequence<Is...> const &)
        -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

    template <std::size_t N>
    using makeIndexSequenceReverse = decltype(indexSequenceReverse(std::make_index_sequence<N>{}));

    template <typename... Ts>
    struct META_ARR;

    template <std::size_t INPUTS>
    struct INPUT;

    template <std::size_t... HIDDENS>
    struct HIDDEN;

    template <std::size_t OUTPUTS>
    struct OUTPUT;

    template <typename float_t, typename... Ts>
    class Layer;

    template <typename float_t, std::size_t OUTPUTS>
    class Layer<float_t, INPUT<OUTPUTS>>
    {
        simd::vector<float_t, OUTPUTS + 1> outputs;

    public:
        Layer()
        {
            for (auto &output : outputs)
            {
                output = 0;
            }
            outputs[OUTPUTS] = 1;
        }
        void load(float_t const input[])
        {
            for (int i = 0; i < OUTPUTS; ++i)
            {
                outputs[i] = input[i];
            }
        }
        auto const &get_outputs() const
        {
            return outputs;
        }
    };

    template <typename float_t, std::size_t INPUTS, std::size_t OUTPUTS>
    class Layer<float_t, HIDDEN<INPUTS>, HIDDEN<OUTPUTS>>
    {
        simd::vector<simd::vector<float_t, INPUTS + 1>, OUTPUTS> weights;
        simd::vector<float_t, OUTPUTS + 1> outputs;
        simd::vector<float_t, OUTPUTS + 1> deltas;

    public:
        static constexpr std::size_t size() { return OUTPUTS; }
        auto const &get_weights() const { return weights; }
        auto const &get_outputs() const { return outputs; }
        auto const &get_deltas() const { return deltas; }

        Layer()
        {
            for (auto &neuron_weights : weights)
            {
                for (auto &weight : neuron_weights)
                {
                    weight = ((float_t)fast_rand()) / ((float_t)FAST_RAND_MAX);
                }
            }
            for (auto &output : outputs)
            {
                output = 0;
            }
            outputs[OUTPUTS] = 1;
        }

        template <typename L>
        void feed(L const &prev_layer)
        {
            auto const &inputs = prev_layer.get_outputs();
            for (int i = 0; i < OUTPUTS; ++i)
            {
                outputs[i] = simd::sum(inputs * weights[i], float_t{0});
                outputs[i] = 1 / (1 + ((float_t)(exp(-outputs[i]))));
            }
        }

        template <typename L1, typename L2>
        void tune(L1 const &prev_layer, L2 const &next_layer, float_t rate)
        {
            auto constexpr next_size = next_layer.size();

            auto const &next_weights = next_layer.get_weights();
            auto const &next_outputs = next_layer.get_outputs();
            auto const &next_deltas = next_layer.get_deltas();
            auto const &inputs = prev_layer.get_outputs();

            for (int i = 0; i < OUTPUTS; ++i)
            {
                deltas[i] = 0;
                for (int j = 0; j < next_size; ++j)
                {
                    deltas[i] += next_deltas[j] * next_weights[j][i];
                }
            }

            deltas = deltas * (outputs * (simd::scalar<decltype(outputs)>(1) - outputs));

            auto const delta_rate = simd::scalar<decltype(deltas)>(rate) * deltas;

            for (int i = 0; i < OUTPUTS; ++i)
            {
                weights[i] = weights[i] + inputs * simd::scalar<typename std::remove_const<typename std::remove_reference<decltype(inputs)>::type>::type>(delta_rate[i]);
            }
        }
    };

    template <typename float_t, std::size_t INPUTS, std::size_t OUTPUTS>
    class Layer<float_t, HIDDEN<INPUTS>, OUTPUT<OUTPUTS>>
    {
        simd::vector<simd::vector<float_t, INPUTS + 1>, OUTPUTS> weights;
        simd::vector<float_t, OUTPUTS> outputs;
        simd::vector<float_t, OUTPUTS> deltas;

    public:
        static constexpr std::size_t size() { return OUTPUTS; }
        auto const &get_weights() const { return weights; }
        auto const &get_outputs() const { return outputs; }
        auto const &get_deltas() const { return deltas; }

        Layer()
        {
            for (auto &neuron_weights : weights)
            {
                for (auto &weight : neuron_weights)
                {
                    weight = ((float_t)fast_rand()) / ((float_t)FAST_RAND_MAX);
                }
            }
            for (auto &output : outputs)
            {
                output = 0;
            }
        }

        template <typename L>
        void feed(L const &prev_layer)
        {
            auto const &inputs = prev_layer.get_outputs();
            for (int i = 0; i < OUTPUTS; ++i)
            {
                outputs[i] = simd::sum(inputs * weights[i], float_t{0});
                outputs[i] = 1 / (1 + ((float_t)(exp(-outputs[i]))));
            }
        }

        template <typename L1, typename L2>
        void tune(L1 const &prev_layer, L2 const &next_layer, float_t rate)
        {
            auto const &inputs = prev_layer.get_outputs();
            auto const &answers = next_layer.get_outputs();

            deltas = (answers - outputs) * (outputs * (simd::scalar<decltype(outputs)>(1) - outputs));
            auto const delta_rate = simd::scalar<decltype(deltas)>(rate) * deltas;

            for (int i = 0; i < OUTPUTS; ++i)
            {
                weights[i] = weights[i] + inputs * simd::scalar<typename std::remove_const<typename std::remove_reference<decltype(inputs)>::type>::type>(delta_rate[i]);
            }
        }
    };

    template <typename float_t, std::size_t OUTPUTS>
    class Layer<float_t, OUTPUT<OUTPUTS>>
    {
        simd::vector<float_t, OUTPUTS> outputs;

    public:
        void load(float_t const answer[])
        {
            for (int i = 0; i < OUTPUTS; ++i)
            {
                outputs[i] = answer[i];
            }
        }
        auto const &get_outputs() const
        {
            return outputs;
        }
    };

    template <typename float_t, typename A, typename B>
    struct MakePerceptronLayers;

    template <typename float_t, typename... As, typename... Bs>
    struct MakePerceptronLayers<float_t, META_ARR<As...>, META_ARR<Bs...>>
    {
        using PerceptronLayers = std::tuple<Layer<float_t, As, Bs>...>;
    };

    template <typename A, typename B, typename C>
    struct JoinLayers;

    template <typename INPUT_LAYER, typename... PERCEPTRONS_LAYERS, typename ANSWER_LAYER>
    struct JoinLayers<INPUT_LAYER, std::tuple<PERCEPTRONS_LAYERS...>, ANSWER_LAYER>
    {
        using Layers = std::tuple<INPUT_LAYER, PERCEPTRONS_LAYERS..., ANSWER_LAYER>;
    };

    template <typename float_t, typename A, typename B, typename C>
    class alignas(32) MLP;

    template <typename float_t, std::size_t INPUTS, std::size_t... HIDDENS, std::size_t OUTPUTS>
    class alignas(32) MLP<float_t, INPUT<INPUTS>, HIDDEN<HIDDENS...>, OUTPUT<OUTPUTS>>
    {
        using InputLayer = Layer<float_t, INPUT<INPUTS>>;
        using PerceptronLayers = typename MakePerceptronLayers<float_t, META_ARR<HIDDEN<INPUTS>, HIDDEN<HIDDENS>...>, META_ARR<HIDDEN<HIDDENS>..., OUTPUT<OUTPUTS>>>::PerceptronLayers;
        using AnswerLayer = Layer<float_t, OUTPUT<OUTPUTS>>;
        using Layers = typename JoinLayers<InputLayer, PerceptronLayers, AnswerLayer>::Layers;

        static constexpr std::size_t INPUT_LAYER = 0;
        static constexpr std::size_t OUTPUT_LAYER = 1 + sizeof...(HIDDENS);
        static constexpr std::size_t ANSWER_LAYER = 1 + sizeof...(HIDDENS) + 1;
        static constexpr std::size_t N_LAYERS = 1 + sizeof...(HIDDENS) + 1 + 1;

        Layers layers;

        template <std::size_t... I>
        void forward(float_t const input[], std::index_sequence<I...>)
        {
            std::get<INPUT_LAYER>(layers).load(input);
            ((std::get<I + 1>(layers).feed(std::get<I>(layers))), ...);
            // 0 1 2 3
        }

        template <std::size_t... I>
        void backprog(float_t const answer[], float_t rate, std::index_sequence<I...>)
        {
            std::get<ANSWER_LAYER>(layers).load(answer);
            ((std::get<I + 1>(layers).tune(std::get<I>(layers), std::get<I + 2>(layers), rate)), ...);
        }

    public:
        void train(float_t const input[], float_t const answer[], float_t rate)
        {
            forward(input, std::make_index_sequence<N_LAYERS - 2>{});
            backprog(answer, rate, makeIndexSequenceReverse<N_LAYERS - 2>{});
        }
        auto const &predict(float_t const input[])
        {
            forward(input, std::make_index_sequence<N_LAYERS - 2>{});
            return std::get<OUTPUT_LAYER>(layers).get_outputs();
        }
    };
};

#endif
