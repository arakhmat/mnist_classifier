#include "mnist_classifier.hpp"

void openDataFile(float** xInput, int* yInput, std::string filename, int loopLimit)
{
	int i = 0, j = 0;

	if(xInput == NULL || yInput == NULL){
        std::cout << "FAILED" << std::endl;
        exit(0);
	}

	std::ifstream file(filename.c_str());
	std::string value;

	if(!file.good())
	{
       std::cout << "FAILED" << std::endl;
       exit(0);
	}

	for(i = 0; i < loopLimit; i++)
	{
            xInput[i] = (float*) malloc(sizeof(float)*(INPUT_SIZE));
            if(xInput[i] == NULL){
                std::cout << "Could not allocate memory for xInput[" << i << "]\n";
                exit(0);
            }
            for(j = 0; j < INPUT_SIZE + 1; j++){
                if(j == 0){
                    getline(file, value, ',');
                    yInput[i] = atoi(value.c_str());
                    //printf("%d\n", yInput[i]);
                }
                else if (j < INPUT_SIZE){
                    getline(file, value, ',');
                    xInput[i][j-1] = atof(value.c_str());
                    xInput[i][j-1] = ((xInput[i][j-1] - 128) / 256);
                    //printf("%f\n", xInput[i][j-1]);
                }
                else {
                    getline(file, value, '\n');
                    xInput[i][j-1] = atof(value.c_str());
                    xInput[i][j-1] = ((xInput[i][j-1] - 128) / 256);
                    //printf("%f\n", xInput[i][j-1]);
                }
            }
	}

	file.close();

}

void initialize3D(float*** toBeInitialized, int numberOfElements, int dimensions, std::string name, int random){
	int i = 0, j = 0, k = 0;

	if(toBeInitialized == NULL){
		printf("Memory allocation for %s failed\r\n", name.c_str());
		exit(0);
	}

	for(i = 0; i < numberOfElements; i++){
		toBeInitialized[i] = (float**) malloc(sizeof(float*)*dimensions);
		if(toBeInitialized[i] == NULL){
			printf("Memory allocation for %s[%d] failed\r\n", name.c_str(), i);
			exit(0);
		}
		for(j = 0; j < dimensions; j++){
			toBeInitialized[i][j] = (float*) malloc(sizeof(float)*dimensions);
			if(toBeInitialized[i][j] == NULL){
                printf("Memory allocation for %s[%d][%d] failed\r\n", name.c_str(), i, j);
                exit(0);
			}
			for(k = 0; k < dimensions; k++){
				if(random == 1)
					toBeInitialized[i][j][k] = (float) (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
				else
					toBeInitialized[i][j][k] = 0;

			}
		}
	}
}

void free3D(float**** toBeFreed, int numberOfElements, int dimensions){
	int i = 0, j = 0;

	for(i = 0; i < numberOfElements; i++){
		for(j = 0; j < dimensions; j++){
			free((*toBeFreed)[i][j]);
		}
		free((*toBeFreed)[i]);
	}
	free(*toBeFreed);
}

void initialize2D(float** toBeInitialized, int numberOfRows, int numberOfColumns, std::string name, int random){
	int i = 0, j = 0;

	if(toBeInitialized == NULL){
		printf("Memory allocation for %s failed\r\n", name.c_str());
		exit(0);
	}

	for(i = 0; i < numberOfRows; i++){
		toBeInitialized[i] = (float*) malloc(sizeof(float)*numberOfColumns);
		if(toBeInitialized[i] == NULL){
			printf("Memory allocation for %s[%d] failed\r\n", name.c_str(), i);
			exit(0);
		}
		for(j = 0; j < numberOfColumns; j++){
			if(random == 1)
				toBeInitialized[i][j] = (float) (LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO))));
			else
				toBeInitialized[i][j] = 0;

		}
	}
}

void free2D(float*** toBeFreed, int numberOfRows){
	int i = 0;

	for(i = 0; i < numberOfRows; i++){
		free((*toBeFreed)[i]);
	}
	free(*toBeFreed);
}

void initialize1D(float* toBeInitialized, int numberOfElements, std::string name, float value){
	int i = 0;

	if(toBeInitialized == NULL){
		printf("Memory allocation for %s failed\r\n", name.c_str());
		exit(0);
	}
	for(i = 0; i < numberOfElements; i++){
		toBeInitialized[i] = value;
	}
}