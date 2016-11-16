#include "mnist_classifier.hpp"


void print3D(float*** toPrint, int numberOfElements, int dimensions, std::string name){
	int i = 0, j = 0, k = 0;

	for(i = 0; i < numberOfElements; i++){
		for(j = 0; j < dimensions; j++){
			for(k = 0; k < dimensions; k++){
				printf("%s is %f at [%d][%d][%d]\r\n", name.c_str(), toPrint[i][j][k], i, j, k);
			}
		}
	}
}
void print3Das3D(float*** toPrint, int numberOfElements, int dimensions, std::string name){
	int i = 0, j = 0, k = 0;

	for(i = 0; i < numberOfElements; i++){
		printf("%s[%d]\n", name.c_str(), i);
		printf("-----------------------------------------------------------------------------------------\n");
		for(j = 0; j < dimensions; j++){
			printf("|");
			for(k = 0; k < dimensions; k++){
				if(toPrint[i][j][k] < 0)
					printf("%f |", toPrint[i][j][k]);
				else
					printf("%f  |", toPrint[i][j][k]);
			}
			printf("\n-----------------------------------------------------------------------------------------\n");
		}
	}
}

void print2D(float** toPrint,  int numberOfRows, int numberOfColumns, std::string name){
	int i = 0, j = 0;

	for(i = 0; i < numberOfRows; i++){
		for(j = 0; j < numberOfColumns; j++){
			printf("%s is %f at [%d][%d]\r\n", name.c_str(), toPrint[i][j], i, j);
		}
	}
}

void print1D(float* toPrint,  int numberOfElements, std::string name){
	int i = 0;

	for(i = 0; i < numberOfElements; i++){
		printf("%s is %f at [%d]\r\n", name.c_str(), toPrint[i], i);
	}
}

void printOutput(float* toPrint,  int numberOfElements){
	int i = 0;

	printf("\r\nOutput: ");
	for(i = 0; i < numberOfElements; i++){
		printf("%f  ", toPrint[i]);
	}
}

void printDigit(float* toPrint,  int numberOfElements){
	int i = 0;

	float value = 0;
	printf("\r\nOutput: ");
	for(i = 0; i < numberOfElements; i++){
		printf("%f  ", toPrint[i]);
		if(toPrint[i] == 1)
			value = i;
	}
	printf(" The value is %f", value);
}
