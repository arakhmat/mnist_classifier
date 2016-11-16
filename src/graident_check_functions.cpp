#include "mnist_classifier.hpp"

void makeEqual2D(float** weightToCopy, int numberOfRows, int numberOfColumns, float** weight){
	int i = 0, j = 0;

	for(i = 0; i < numberOfRows; i++){
		for(j = 0; j < numberOfColumns; j++){
			weight[i][j] = weightToCopy[i][j];
		}
	}
}


void makeEqual3D(float*** weightToCopy, int numberOfElements, int dimensions, float*** weight){
	int i = 0, j = 0, k = 0;

	for(i = 0; i < numberOfElements; i++){
		for(j = 0; j < dimensions; j++){
			for(k = 0; k < dimensions; k++){
				weight[i][j][k] = weightToCopy[i][j][k];
			}
		}
	}
}
