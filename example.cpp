#include "mlp.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <utility>
#include <time.h>

#define CHECK_TIME(x)                                                                                       \
    {                                                                                                       \
        struct timespec start, end;                                                                         \
        clock_gettime(CLOCK_REALTIME, &start);                                                              \
        x;                                                                                                  \
        clock_gettime(CLOCK_REALTIME, &end);                                                                \
        double f = ((double)end.tv_sec * 1e9 + end.tv_nsec) - ((double)start.tv_sec * 1e9 + start.tv_nsec); \
        printf("time %f ms\n", f / 1000000);                                                                \
    }

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&...args)
{
    TimeVar t1 = timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow() - t1);
}

int main(int argc, char **argv);
void readIris();
void shuffle(int *array, int n);

#define epochs 100000
#define learning_rate 0.1
#define rand_seed 0
#define cols 4
#define out_cols 3
#define rows 150
#define train_rows 105
#define n_layers 3

namespace mai = meta_ai;

mai::MLP<float, mai::INPUT<cols>, mai::HIDDEN<7, 3>, mai::OUTPUT<out_cols>> mlp;

int order[rows];
float feat[rows * cols];
float label[rows * out_cols];

int main(int argc, char **argv)
{
    srand(rand_seed);
    readIris();

    for (int i = 0; i < rows; ++i)
        order[i] = i;
    shuffle(order, rows);

    CHECK_TIME(
        for (int i = 0; i < epochs; i++) {
            shuffle(order, train_rows);
            for (int j = 0; j < train_rows; j++)
            {
                int row = order[j];
                mlp.train(feat + row * cols, label + row * out_cols, learning_rate);
            }
        })

    int correct = 0;
    int incorrect = 0;

    for (int i = train_rows; i < rows; ++i)
    {
        int row = order[i];
        // printf("ROW: %d\n", row);

        auto const &prediction = mlp.predict(feat + row * cols);

        if (abs(prediction[0] - label[row * out_cols + 0]) < 0.1 &&
            abs(prediction[1] - label[row * out_cols + 1]) < 0.1 &&
            abs(prediction[2] - label[row * out_cols + 2]) < 0.1)
            correct++;
        else
            incorrect++;
        // printf("Predicted: [%f, %f, %f] vs Answer: [ %f, %f, %f ]\n\n", prediction[0],
        //        prediction[1], prediction[2], label[row * out_cols + 0],
        //       label[row * out_cols + 1], label[row * out_cols + 2]);
    }

    printf("Total de predicciones correctas: %d\n", correct);
    printf("Total de predicciones incorrectas: %d\n", incorrect);

    return EXIT_SUCCESS;
}

void readIris()
{

    char const *const dataFileName = "iris.data";

    memset(feat, 0, rows * cols * sizeof(float));
    memset(label, 0, rows * out_cols * sizeof(float));

    printf("Obsns size is %d and feat size is %d.\n", rows, cols);

    FILE *fpDataFile = fopen(dataFileName, "r");

    if (!fpDataFile)
    {
        printf("Missing input file: %s\n", dataFileName);
        exit(1);
    }

    int index = 0;
    char line[1024];
    char flowerType[20];
    float l;

    while (fgets(line, 1024, fpDataFile))
    {
        if (5 == sscanf(line, "%f,%f,%f,%f,%f[^\n]", &feat[index * cols + 0],
                        &feat[index * cols + 1], &feat[index * cols + 2],
                        &feat[index * cols + 3], &l))
        {
            // printf("%f,%f,%f,%f,%f\n", feat[index * cols + 0], feat[index * cols + 1],
            //        feat[index * cols + 2], feat[index * cols + 3], l);
            label[index * out_cols + ((int)l)] = 1;
            index++;
        }
    }
    fclose(fpDataFile);
}

void shuffle(int *array, int n)
{
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}