#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <sys/types.h>

#define matrix_set_element(mat, x, y, val) internal_matrix_set_element(mat, x, y, val, #mat, #x, #y, #val, __FILE__, __LINE__)
#define matrix_get_element(mat, x, y) internal_matrix_get_element(mat, x, y, #mat, #x, #y, __FILE__, __LINE__)
#define matrix_dot(matA, matB) internal_matrix_dot(matA, matB, #matA, #matB, __FILE__, __LINE__)
#define matrix_bitwise_operator(matA, matB, operator) internal_matrix_bitwise_operator(matA, matB, operator, #matA, #matB, #operator, __FILE__, __LINE__)

typedef struct {
  size_t height;
  size_t width;
  double *data;
} internal_matrix_t;

typedef void* matrix_t;

matrix_t matrix_create(size_t width, size_t height);

void matrix_delete(matrix_t mat);

void matrix_set(matrix_t mat, double val);

void internal_matrix_set_element(matrix_t mat, size_t x, size_t y, double val, char* matName, char* xName, char* yName, char* valName, char * file, unsigned long line);

double internal_matrix_get_element(matrix_t mat, size_t x, size_t y, char* matName, char* xName, char* yName, char* file, unsigned long line);

matrix_t internal_matrix_dot(matrix_t matA, matrix_t matB, char* matAName, char* matBName, char* file, unsigned long line);

size_t matrix_get_width(matrix_t mat);

size_t matrix_get_height(matrix_t mat);

void matrix_print(matrix_t mat);

matrix_t matrix_transform(matrix_t mat, double(*transFun)(double));

matrix_t internal_matrix_bitwise_operator(matrix_t matA, matrix_t matB, double(*op)(double,double), char* matAName, char* matBName, char* opName, char* file, unsigned long line);

matrix_t matrix_transpose(matrix_t mat);

matrix_t matrix_duplicate(matrix_t mat);

matrix_t matrix_create_row_from_array(const double *data, size_t len);

matrix_t matrix_create_column_from_array(const double *data, size_t len);

#endif//_MATRIX_H_
