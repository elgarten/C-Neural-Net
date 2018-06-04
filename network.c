#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>

#include "network.h"
#include "matrix.h"

network_t network_create(const size_t *layerSizes, size_t layers){
  internal_network_t *ret = (internal_network_t *) malloc(sizeof(internal_network_t));
  ret->size = layers - 1;
  ret->layers = (internal_layer_t *) malloc((layers - 1) * sizeof(internal_layer_t));
  for(size_t i = 1; i < layers; i++){
    ret->layers[i - 1].weights = matrix_create(layerSizes[i - 1], layerSizes[i]);
    ret->layers[i - 1].bias = matrix_create(1, layerSizes[i]);
    matrix_set(ret->layers[i - 1].weights, 1);
    matrix_set(ret->layers[i - 1].bias, 1);
  }
  return (network_t) ret;
}

void network_delete(network_t net){
  internal_network_t *intNet = (internal_network_t *) net;
  for(size_t i = 0; i < intNet->size; i++){
    matrix_delete(intNet->layers[i].weights);
    matrix_delete(intNet->layers[i].bias);
  }
  free(intNet->layers);
  free(intNet);
}

void network_randomize(network_t net){
  internal_network_t *intNet = (internal_network_t *) net;
  matrix_t temp;
  for(size_t layerIndex = 0; layerIndex < intNet->size; layerIndex++){
    temp = intNet->layers[layerIndex].weights;
    intNet->layers[layerIndex].weights = matrix_transform(temp, randNum);
    matrix_delete(temp);
    temp = intNet->layers[layerIndex].bias;
    intNet->layers[layerIndex].bias = matrix_transform(temp, randNum);
    matrix_delete(temp);
  }
}

matrix_t network_feedforward(network_t net, matrix_t input){
  internal_network_t *intNet = (internal_network_t *) net;
  matrix_t weightResult;
  matrix_t biasResult;
  matrix_t sigmoidResult = matrix_duplicate(input);
  for(size_t layerIndex = 0; layerIndex < intNet->size; layerIndex++){
    weightResult = matrix_dot(intNet->layers[layerIndex].weights,sigmoidResult);
    matrix_delete(sigmoidResult);
    biasResult = matrix_bitwise_operator(weightResult,intNet->layers[layerIndex].bias,add);
    matrix_delete(weightResult);
    sigmoidResult = matrix_transform(biasResult, sigmoid);
    matrix_delete(biasResult);
  }
  return sigmoidResult;
}

void randInit(){
    time_t t;
    srand((unsigned) time(&t));
}

int randRange(int min, int max){
  return (min + (rand() % (max - min)));
}

double randNum(double a){
  (void)(a);
  return (double)(randRange(-100000, 100000))/10000;
}

double add(double a, double b){ return a + b;}

void network_print(network_t net){
  internal_network_t *intNet = (internal_network_t *) net;
  for(size_t layerIndex =  0; layerIndex < intNet->size; layerIndex++){
    printf("Layer %lu Weights\n", layerIndex + 1);
    matrix_print(intNet->layers[layerIndex].weights);
    printf("-----------------------------------------------------------\n");
    printf("Layer %lu Bias\n", layerIndex + 1);
    matrix_print(intNet->layers[layerIndex].bias);
    printf("-----------------------------------------------------------\n");
  }
}

double sigmoid(double a){
  return (1.0 / (1.0 + exp(-1 * a)));
}

double sigmoid_derivative(double a){
  return sigmoid(a) * (1 - sigmoid(a));
}

void network_backprop(network_t net, matrix_t input, matrix_t output, matrix_t **nablaWLocation, matrix_t **nablaBLocation){
	internal_network_t *intNet = (internal_network_t *) net;

	matrix_t *nablaW = calloc(intNet->size, sizeof(matrix_t));
	matrix_t *nablaB = calloc(intNet->size, sizeof(matrix_t));
	*nablaWLocation = nablaW;
	*nablaBLocation = nablaB;

	matrix_t *activations = calloc(intNet->size + 1, sizeof(matrix_t));

	activations[0] = input;	
	matrix_t *zs = calloc(intNet->size, sizeof(matrix_t));

	matrix_t weightResult;
	for(size_t i = 0; i < intNet->size; ++i){
		weightResult = matrix_dot(intNet->layers[i].weights, activations[i]);
		zs[i] = matrix_bitwise_operator(weightResult, intNet->layers[i].bias, add);
		activations[i + 1] = matrix_transform(zs[i], sigmoid);
		matrix_delete(weightResult);
	}
	
	matrix_t costDerivativeResults = matrix_bitwise_operator(activations[intNet->size], output, cost_derivative);
	matrix_t sigmoidDerivativeResult = matrix_transform(zs[intNet->size - 1], sigmoid_derivative);
	nablaB[intNet->size - 1] = matrix_bitwise_operator(costDerivativeResults, sigmoidDerivativeResult, multiply);
	matrix_delete(costDerivativeResults);
	matrix_delete(sigmoidDerivativeResult);
	
	matrix_t transposedActivations = matrix_transpose(activations[intNet->size - 1]);
	nablaW[intNet->size - 1] = matrix_dot(nablaB[intNet->size - 1], transposedActivations);
	matrix_delete(transposedActivations);

	matrix_t transposedWeights;
	matrix_t dotResult;
	for(size_t i = 2; i <= intNet->size; ++i){
		sigmoidDerivativeResult = matrix_transform(zs[intNet->size - i], sigmoid_derivative);
		transposedWeights = matrix_transpose(intNet->layers[(intNet->size - i) + 1].weights);
		dotResult = matrix_dot(transposedWeights, nablaB[(intNet->size - i) + 1]);
		nablaB[intNet->size - i] = matrix_bitwise_operator(dotResult, sigmoidDerivativeResult, multiply);	
		transposedActivations = matrix_transpose(activations[(intNet->size - i)]);
		nablaW[intNet->size - i] = matrix_dot(nablaB[intNet->size - i], transposedActivations);
		matrix_delete(sigmoidDerivativeResult);
		matrix_delete(transposedWeights);
		matrix_delete(dotResult);
		matrix_delete(transposedActivations);
	}
	
	/*
	 * Variables To Deallocate
	 * activations matrix_t[intNet->size + 1]
	 * zs matrix_t[intNet->size]
	 */
	for(size_t i = 0; i < intNet->size; i++){
		matrix_delete(activations[i + 1]);
		matrix_delete(zs[i]);
	}
}

double cost(double activation, double expected){
  return 0.5 * (activation - expected) * (activation - expected);
}

double cost_derivative(double activation, double expected){
  return activation - expected;
}

double multiply(double a, double b){
  return a * b;
}

void network_train(network_t net, matrix_t input, matrix_t output, double learning_rate){ 
	internal_network_t *intNet = (internal_network_t *) net;
	matrix_t *nabla_w, *nabla_b;
	network_backprop(net, input, output, &nabla_w, &nabla_b);
  matrix_t layer_nabla_w;
  matrix_t layer_nabla_b;
  matrix_t learning_rate_matrix_w;
  matrix_t learning_rate_matrix_b;
  matrix_t temp_weights;
  matrix_t temp_bias;
  learning_rate *= -1;
  for(size_t layerIndex = 0; layerIndex < intNet->size; layerIndex++){
      learning_rate_matrix_w = matrix_create(matrix_get_width(nabla_w[layerIndex]),matrix_get_height(nabla_w[layerIndex]));
      matrix_set(learning_rate_matrix_w, learning_rate);
      learning_rate_matrix_b = matrix_create(matrix_get_width(nabla_b[layerIndex]),matrix_get_height(nabla_b[layerIndex]));
      matrix_set(learning_rate_matrix_b, learning_rate);
      layer_nabla_w = matrix_bitwise_operator(nabla_w[layerIndex], learning_rate_matrix_w,multiply);
      layer_nabla_b = matrix_bitwise_operator(nabla_b[layerIndex], learning_rate_matrix_b,multiply);
      temp_weights = matrix_bitwise_operator(intNet->layers[layerIndex].weights,layer_nabla_w,add);
      temp_bias = matrix_bitwise_operator(intNet->layers[layerIndex].bias,layer_nabla_b,add);
      matrix_delete(intNet->layers[layerIndex].weights);
      matrix_delete(intNet->layers[layerIndex].bias);
      intNet->layers[layerIndex].weights = temp_weights;
      intNet->layers[layerIndex].bias = temp_bias;
      matrix_delete(layer_nabla_w);
      matrix_delete(layer_nabla_b);
      matrix_delete(learning_rate_matrix_w);
      matrix_delete(learning_rate_matrix_b);
      matrix_delete(nabla_w[layerIndex]);
      matrix_delete(nabla_b[layerIndex]);
  }
	free(nabla_w);
	free(nabla_b);
}
