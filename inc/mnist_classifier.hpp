#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cmath>


const float LEARNING_RATE = 0.5f;
const float BIAS_LEARNING_RATE = 0.01;

const int INPUT_SIZE = 784;

const int SIZE_OF_BATCH = 50;

const int NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1 = 5;
const int NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2 = 20;

const int NFM_1 = NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1;
const int NFM_2 = NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2;

const int NUMBER_OF_KERNELS_CONV_LAYER_1 = NFM_1;
const int NUMBER_OF_KERNELS_CONV_LAYER_2 = (NFM_1 * NFM_2);

const int NK_1 = NUMBER_OF_KERNELS_CONV_LAYER_1;
const int NK_2 = NUMBER_OF_KERNELS_CONV_LAYER_2;

const int DIMENSIONS_OF_KERNEL = 5;		//Kernel is 5*5
const int DIMENSIONS_OF_POOLING = 2;		//Pooling kernel is 2*2

const int DK = DIMENSIONS_OF_KERNEL;
const int DP = DIMENSIONS_OF_POOLING;

const int DIMENSIONS_OF_IMAGE = 28;
const int DIMENSIONS_OF_FEATURE_MAP_1 = (DIMENSIONS_OF_IMAGE - DK + 1);
const int DIMENSIONS_OF_POOLED_MAP_1 = (DIMENSIONS_OF_FEATURE_MAP_1 / DP);
const int DIMENSIONS_OF_FEATURE_MAP_2 = (DIMENSIONS_OF_POOLED_MAP_1 - DK + 1);
const int DIMENSIONS_OF_POOLED_MAP_2 = (DIMENSIONS_OF_FEATURE_MAP_2 / DP);

const int DI = DIMENSIONS_OF_IMAGE;
const int DFM_1 = DIMENSIONS_OF_FEATURE_MAP_1;
const int DPM_1 = DIMENSIONS_OF_POOLED_MAP_1;
const int DFM_2 = DIMENSIONS_OF_FEATURE_MAP_2;
const int DPM_2 = DIMENSIONS_OF_POOLED_MAP_2;

const int SIZE_OF_POOLED_MAP_2 = (DPM_2 * DPM_2);

const int NUMBER_OF_INPUTS_HIDDEN_LAYER = (NFM_2 * SIZE_OF_POOLED_MAP_2);
const int NUMBER_OF_OUTPUTS_HIDDEN_LAYER = 100;
const int NUMBER_OF_CLASSES = 10;

const int NUMBER_OF_EXAMPLES = 60000;
const int NUMBER_OF_TEST_EXAMPLES = 10000;

const float RANDOM_VALUES_HI = 0.5;
const float RANDOM_VALUES_LO = -0.5;

const float HI = RANDOM_VALUES_HI;
const float LO = RANDOM_VALUES_LO;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// neural_network_functions.c
void convLayer(float*** input, float* bias, float*** kernel, float (*nonLinFunc)(float z),
		int numberOfInputs, int kernelRows,  int kernelColumns,
		int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
		float*** featureMap);
void poolLayer(float*** featureMap, int numberOfFeatureMaps,
		int featureMapRows, int featureMapColumns,
		int poolingRows, int poolingColumns, float*** maxPooledMap);
void fulConLayer(float* input, float bias, float** weight, float (*nonLinFunc)(float z),
		int sizeOfOutput, int sizeOfInput, float* output);

float costFunction(float* trueValue, float* calculatedValue, int numberOfClasses);

float LFCLBackprop(float* trueValue, float* calculatedValue,  float* input, float bias,
		int sizeOfLayer, int sizeOfNextLayer, float* error, float** gradient);

float FCLBackprop(float* prevLayerError, float** prevTheta, float** currentTheta, float* input, float bias,
		 float (*nonLinFuncDer)(float z), int sizeOfLayer, int sizeOfPrevLayer, int sizeOfNextLayer,
		 float* error, float** gradient);

void poolFromFCLBackprop(float* prevLayerError, float** prevTheta,
		int sizeOfLayer, int sizeOfPrevLayer, float* error);

void upSampleBackprop(float*** poolingError, float*** featureMap,
		int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
		int poolingRows, int poolingColumns, float*** convError);
void convLayerSigGrad(float*** input, float* bias, float*** kernel, float (*nonLinFuncDer)(float z),
		int numberOfInputs, int kernelRows,  int kernelColumns,
		int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
		float*** featureMap);
void elWiseMult3D(float*** var1, float*** var2, int numberOfFeatureMaps, int featureMapRows, int featureMapColumns);
void convGrad(float*** input, float*** errorMap,
		int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
		int numberOfKernels, int kernelRows, int kernelColumns, float*** kernelGradient);
void accumConvBiasGrad(float*** error, float* biasGradAcc, int numberOfFeatureMaps, int featureMapRows, int featureMapColumns);
void convBackprop(float*** errorIn, float*** kernel,
		int numberOfErrorIns, int errorInRows, int errorInColumns,
		int numberOfKernels, int kernelRows,  int kernelColumns,
		int numberOfErrorOuts, int errorOutRows, int errorOutColumns, float*** errorOut);

void updateGrad3D(float*** weight, float*** gradientAcc, int numberOfElements,
		int numberOfRows, int numberOfColumns, float learningRate, int sizeOfBatch);
void accumGrad3D(float*** gradientAcc, float*** gradient, int numberOfElements, int numberOfRows, int numberOfColumns);
void updateGrad2D(float** weight, float** gradientAcc, int numberOfRows, int numberOfColumns, float learningRate,
		int sizeOfBatch);

void accumGrad2D(float** gradientAcc, float** gradient, int numberOfRows, int numberOfColumns);
void updateGrad1D(float* toUpdate, float* gradientAcc, int numberOfElements, float learningRate, int sizeOfBatch);

void updateGrad(float toUpdate, float gradientAcc, float learningRate, int sizeOfBatch);

float sigmoid(float z);
float sigmoidGradient(float z);

void digitInit(float* digit, int y, int numberOfClasses);
void imageInit(float*** image, float* x, int imageRows, int imageColumns);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// transform_functions.c
void transform3Dto1D(float*** in, int numberOfElements, int numberOfColumns, int numberOfRows, float* out);
void transform1Dto3D(float* in, int numberOfElements, int numberOfColumns, int numberOfRows, float*** out);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// initialize_functions.c
void openDataFile(float** xInput, int* yInput, std::string filename, int loopLimit);
void initialize1D(float* toBeInitialized, int numberOfElements, std::string name, float value);
void initialize2D(float** toBeInitialized, int numberOfRows, int numberOfColumns, std::string name, int random);
void initialize3D(float*** toBeInitialized, int numberOfElements, int dimensions, std::string name, int random);

void free2D(float*** toBeFreed, int numberOfRows);
void free3D(float**** toBeFreed, int numberOfElements, int dimensions);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// print_functions.c
void printOutput(float* toPrint,  int numberOfElements);
void printDigit(float* toPrint,  int numberOfElements);
void print1D(float* toPrint,  int numberOfElements, std::string name);
void print2D(float** toPrint,  int numberOfRows, int numberOfColumns, std::string name);
void print3D(float*** toPrint, int numberOfElements, int dimensions, std::string name);
void print3Das3D(float*** toPrint, int numberOfElements, int dimensions, std::string name);
// gradient_check_functions.c
void makeEqual2D(float** weightToCopy, int numberOfRows, int numberOfColumns, float** weight);
void makeEqual3D(float*** weightToCopy, int numberOfElements, int dimensions, float*** weight);
