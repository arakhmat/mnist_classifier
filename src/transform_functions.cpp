#include "mnist_classifier.hpp"

void transform3Dto1D(float*** in, int numberOfElements, int numberOfColumns, int numberOfRows, float* out)
{
	int i = 0, j = 0, k = 0;

	for(i = 0; i < numberOfElements; i++){
		for(j = 0; j < numberOfRows; j++){
			for(k = 0; k < numberOfColumns; k++){
				out[(i * numberOfRows + j) * numberOfColumns + k] = in[i][j][k];
			}
		}
	}
}

void transform1Dto3D(float* in, int numberOfElements, int numberOfColumns, int numberOfRows, float*** out)
{
	int i = 0, j = 0, k = 0;

	for(i = 0; i < numberOfElements; i++){
		for(j = 0; j < numberOfRows; j++){
			for(k = 0; k < numberOfColumns; k++){
				out[i][j][k] = in[(i * numberOfRows + j) * numberOfColumns + k];
			}
		}
	}
}
