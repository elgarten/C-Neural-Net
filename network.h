#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "matrix.h"

typedef struct {
  matrix_t weights;
  matrix_t bias;
} internal_layer_t;

typedef struct {
  size_t size;
  internal_layer_t *layers;
} internal_network_t;

typedef void* network_t;

network_t network_create(const size_t *layerSizes, size_t layers);

void network_delete(network_t net);

void network_randomize(network_t net);

matrix_t network_feedforward(network_t net, matrix_t input);

void randInit();

int randRange(int min, int max);

double randNum(double a);

double add(double a, double b);

void network_print(network_t net);

double sigmoid(double a);

double sigmoid_prime(double a);

void backprop(network_t net, matrix_t input, matrix_t output, matrix_t **nablaWLocation, matrix_t **nablaBLocation);

double cost(double activation, double expected);

double cost_derivative(double activation, double expected);

double multiply(double a, double b);

void network_train(network_t net, matrix_t input, matrix_t output, double learning_rate);

#endif//_NETWORK_H_
