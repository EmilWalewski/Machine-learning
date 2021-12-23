#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void fillMatrixRandomValues(int* matrix, int rows, int columns);
void printMatrix(int* matrix, int rows, int columns);
void multiplicateMatrixes(int* mat1, int* mat2, int* result, int rows, int columns, int mat1Size, int mat2Size);

int main(int argc, char** argv){

srand (time(NULL));

int rows = 3;
int cols = 3;

int rows2 = 3;
int cols2 = 3;

int *mat1 = (int *)malloc(rows * cols * sizeof(int));
int *mat2 = (int *)malloc(rows2 * cols2 * sizeof(int));
int *result = (int *)malloc(rows * cols * sizeof(int));

fillMatrixRandomValues(mat1, rows, cols);
fillMatrixRandomValues(mat2, rows2, cols2);

printMatrix(mat1, rows, cols);
printf("\n");
printMatrix(mat2, rows2, cols2);
printf("\n");
printf("Result: \n");
multiplicateMatrixes(mat1, mat2, result, rows, cols, rows*cols, rows2*cols2);
printMatrix(result, rows, cols);


free(mat1);
free(mat2);
free(result);


return 0;
}

void multiplicateMatrixes(int* mat1, int* mat2, int* result, int rows, int columns, int mat1Size, int mat2Size){
    if(mat1Size > mat2Size){
        rows = rows - 1;
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            result[i * columns + j] = 0;
            for(int k = 0; k < columns; k++){
                if(mat2Size >= mat1Size){
                    result[i * columns + j] += mat1[i * columns + k] * mat2[k * rows + j];
                }else{
                    result[i * columns + j] += mat2[i * columns + k] * mat1[k * rows + j];
                }
            }
        }
    }
}

void fillMatrixRandomValues(int* matrix, int rows, int columns){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            matrix[i * columns + j] = rand() % 9 + 1;
        }
    }
}

void printMatrix(int* matrix, int rows, int columns){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            printf("%d ", matrix[i * columns + j]);
        }
        printf("\n");
    }
}