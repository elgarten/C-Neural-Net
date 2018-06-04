#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <string.h>

#include "matrix.h"

matrix_t matrix_create(size_t width, size_t height){
  internal_matrix_t *mat = (internal_matrix_t *) malloc(sizeof(internal_matrix_t));
  mat->data = (double *)calloc(width * height, sizeof(double));
  mat->width = width;
  mat->height = height;
  return (matrix_t) mat;
}

void matrix_delete(matrix_t mat){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  free(intMat->data);
  free(intMat);
}

void matrix_set(matrix_t mat, double val){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  for(size_t i = 0; i < intMat->width * intMat->height; i++){
    intMat->data[i] = val;
  }
}

void internal_matrix_set_element(matrix_t mat, size_t x, size_t y, double val, char* matName, char* xName, char* yName, char* valName, char * file, unsigned long line){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  if(x > (intMat->width - 1) || y > (intMat->height - 1)){
    fprintf(stderr, "Matrix Error: \n\tIn File: %s Line: %lu \n\tmatrix_set_element(%s, %s, %s,%s) \n\tIndex Out Of Bounds: Attempted To Write To Position (%lu, %lu), But Matrix Is Of Size (%lu, %lu)\n",
    file, line, matName, xName, yName, valName, x, y, intMat->width, intMat->height);
    exit(1);
  }
  const size_t index = ((y * intMat->width) + x);
  intMat->data[index] = val;
}

double internal_matrix_get_element(matrix_t mat, size_t x, size_t y, char* matName, char* xName, char* yName, char* file, unsigned long line){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  if(x > (intMat->width - 1) || y > (intMat->height - 1)){
    fprintf(stderr, "Matrix Error: \n\tIn File: %s Line: %lu \n\tmatrix_get_element(%s, %s, %s) \n\tIndex Out Of Bounds: Attempted To Write To Position (%lu, %lu), But Matrix Is Of Size (%lu, %lu)\n",
    file, line, matName, xName, yName, x, y, intMat->width, intMat->height);
    exit(1);
  }
  const size_t index = ((y * intMat->width) + x);
  return intMat->data[index];
}

matrix_t internal_matrix_dot(matrix_t matA, matrix_t matB, char* matAName, char* matBName, char* file, unsigned long line){
  internal_matrix_t *intMatA = (internal_matrix_t *) matA;
  internal_matrix_t *intMatB = (internal_matrix_t *) matB;
  if(intMatA->width != intMatB->height){
    fprintf(stderr, "Matrix Error: \n\tIn File: %s Line: %lu \n\tmatrix_dot(%s,%s) \n\tBad Dimensions: Matricies are of Dimensions (%lu, %lu) and (%lu, %lu)\n",
    file, line, matAName, matBName, intMatA->width, intMatA->height, intMatB->width, intMatB->height);
    exit(1);
  }
  matrix_t ret = matrix_create(intMatB->width, intMatA->height);
  for(size_t y = 0; y < matrix_get_height(ret); y++){
    for(size_t x = 0; x < matrix_get_width(ret); x++){
      double sum = 0;
      for(size_t i = 0; i < intMatA->width; i++){
        sum += matrix_get_element(matA, i, y) * matrix_get_element(matB, x, i);
      }
      matrix_set_element(ret, x, y, sum);
    }
  }
  return ret;
}

inline size_t matrix_get_width(matrix_t mat){
  return ((internal_matrix_t *) mat)->width;
}

inline size_t matrix_get_height(matrix_t mat){
  return ((internal_matrix_t *) mat)->height;
}

void matrix_print(matrix_t mat){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  for(size_t y = 0; y < intMat->height; y++){
    for(size_t x = 0; x < intMat->width; x++){
      printf("%4.4f\t", intMat->data[y * intMat->width + x]);
    }
    printf("\n");
  }
}

matrix_t matrix_transform(matrix_t mat, double(*transFun)(double)){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  internal_matrix_t *ret = (internal_matrix_t *) matrix_create(intMat->width, intMat->height);
  for(size_t y = 0; y < intMat->height; y++){
    for(size_t x = 0; x < intMat->width; x++){
      ret->data[y * intMat->width + x] = transFun(intMat->data[y * intMat->width + x]);
    }
  }
  return (matrix_t) ret;
}

matrix_t internal_matrix_bitwise_operator(matrix_t matA, matrix_t matB, double(*operator)(double,double), char* matAName, char* matBName, char* operatorName, char* file, unsigned long line){
  internal_matrix_t *intMatA = (internal_matrix_t *) matA;
  internal_matrix_t *intMatB = (internal_matrix_t *) matB;
  if(intMatA->width != intMatB->width || intMatA->height != intMatB->height){
    fprintf(stderr, "Matrix Error: \n\tIn File: %s Line: %lu \n\tmatrix_bitwise_operator(%s, %s, %s)\n\tBad Dimensions: Matricies Are Of Dimensions (%lu, %lu) and (%lu, %lu)",
            file, line, matAName, matBName, operatorName,intMatA->width, intMatA->height, intMatB->width, intMatB->height);
    exit(1);
  }
  internal_matrix_t *ret = (internal_matrix_t *) matrix_create(intMatA->width, intMatA->height);

  for(size_t y = 0; y < ret->height; y++){
    for(size_t x = 0; x < ret->width; x++){
      ret->data[y * ret->width + x] = operator(intMatA->data[y * ret->width + x], intMatB->data[y * ret->width + x]);
    }
  }
  return (matrix_t) ret;
}

matrix_t matrix_transpose(matrix_t mat){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  matrix_t ret = matrix_create(intMat->height, intMat->width);
  for(size_t y = 0; y < intMat->height; y++){
    for(size_t x = 0; x < intMat->width; x++){
      matrix_set_element(ret, y, x, matrix_get_element(mat, x, y));
    }
  }
  return ret;
}

matrix_t matrix_duplicate(matrix_t mat){
  internal_matrix_t *intMat = (internal_matrix_t *) mat;
  internal_matrix_t *ret = (internal_matrix_t *) matrix_create(intMat->width,intMat->height);
  memcpy(ret->data,intMat->data,ret->width*ret->height*sizeof(double));
  return (matrix_t) ret;
}

matrix_t matrix_create_row_from_array(const double *data, size_t len){
	internal_matrix_t *intMat = (internal_matrix_t *) matrix_create(len, 1);
	for(size_t i = 0; i < len; ++i){
		intMat->data[i] = data[i];
	}
	return (matrix_t) intMat;
}

matrix_t matrix_create_column_from_array(const double *data, size_t len){
	internal_matrix_t *intMat = (internal_matrix_t *) matrix_create(1, len);
	for(size_t i = 0; i < len; ++i){
		intMat->data[i] = data[i];
	}
	return (matrix_t) intMat;
}
