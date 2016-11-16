#include "mnist_classifier.hpp"

using namespace std;

int main(){

	printf("Training:\n");
	int i = 0, j = 0;

	float** x = (float**) malloc(sizeof(float*)*NUMBER_OF_EXAMPLES);
	int* y = (int*) malloc(sizeof(int)*NUMBER_OF_EXAMPLES);

	float cost = 0;

	int success = 0;
	int result = 0;
	float val = 0;

	float*** image = (float***) malloc(sizeof(float**)*1);

	float*** kernel_1 = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_1);
	float*** kernel_2 = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_2);

	float*** featureMap_1 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);
	float*** featureMap_2 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);
	float*** maxPooledMap_1 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);
	float*** maxPooledMap_2 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);
	float* maxPooledMap_2_1D = (float*) malloc(sizeof(float)*NUMBER_OF_INPUTS_HIDDEN_LAYER);
	float* hiddenLayerOutput = (float*) malloc(sizeof(float)*NUMBER_OF_OUTPUTS_HIDDEN_LAYER);
	float* output = (float*) malloc(sizeof(float) * NUMBER_OF_CLASSES);

	float** weight1 = (float**) malloc(sizeof(float*) * NUMBER_OF_OUTPUTS_HIDDEN_LAYER);
	float** weight2 = (float**) malloc(sizeof(float*) * NUMBER_OF_CLASSES);

	float* digit = (float*) malloc(sizeof(float) * NUMBER_OF_CLASSES);

	float* errorLFCL = (float*) malloc(sizeof(float) * NUMBER_OF_CLASSES);
	float** gradientLFCL = (float**) malloc(sizeof(float*) * NUMBER_OF_CLASSES);

	float* errorHL = (float*) malloc(sizeof(float) * NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 4);
	float** gradientHL = (float**) malloc(sizeof(float*) * NUMBER_OF_OUTPUTS_HIDDEN_LAYER);

	float* errorPL2 = (float*) malloc(sizeof(float) * NUMBER_OF_INPUTS_HIDDEN_LAYER + 4);
	float*** errorPL2_3D = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);

	float*** errorCL2 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);
	float*** kernelGradient_2 = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_2);

	float*** errorPL1 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);

	float*** errorCL1 = (float***) malloc(sizeof(float**)*NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);

	float*** kernelGradient_1 = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_1);

	float** gradientLFCL_Acc = (float**) malloc(sizeof(float*) * NUMBER_OF_CLASSES);
	float** gradientHL_Acc = (float**) malloc(sizeof(float*) * NUMBER_OF_OUTPUTS_HIDDEN_LAYER);
	float*** kernelGradient_2_Acc = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_2);
	float*** kernelGradient_1_Acc = (float***) malloc(sizeof(float**)*NUMBER_OF_KERNELS_CONV_LAYER_1);

	float* biasOfConv_1 = (float*) malloc(sizeof(float) * NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);
	float* biasOfConv_2 = (float*) malloc(sizeof(float) * NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);
	float biasHidden = 1;
	float biasLast = 1;

	float* biasOfConvGradAcc_1 = (float*) malloc(sizeof(float) * NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1);
	float* biasOfConvGradAcc_2 = (float*) malloc(sizeof(float) * NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2);
	float biasHiddenGradAcc = 0;
	float biasLastGradAcc = 0;

	srand (static_cast <unsigned> (time(0)));

	initialize3D(image, 1, 28, "image", 0);

	initialize3D(kernel_1, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, "kernel_1", 1);
	initialize3D(kernel_2, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, "kernel_2", 1);

	initialize3D(featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1,
			DIMENSIONS_OF_FEATURE_MAP_1, "featureMap_1", 0);
	initialize3D(featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
			DIMENSIONS_OF_FEATURE_MAP_2, "featureMap_2", 0);
	initialize3D(maxPooledMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1,
			DIMENSIONS_OF_POOLED_MAP_1, "maxPooledMap_1", 0);
	initialize3D(maxPooledMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
			DIMENSIONS_OF_POOLED_MAP_2, "maxPooledMap_2", 0);

	initialize1D(maxPooledMap_2_1D, NUMBER_OF_INPUTS_HIDDEN_LAYER, "maxPooledMap_2_1D", 0);
	initialize1D(hiddenLayerOutput, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, "hiddenLayerOutput", 0);
	initialize1D(output, NUMBER_OF_CLASSES, "output", 0);

	initialize2D(weight1, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, NUMBER_OF_INPUTS_HIDDEN_LAYER + 1, "weight1", 1);
	initialize2D(weight2, NUMBER_OF_CLASSES, NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1, "weight2", 1);

	initialize1D(errorLFCL, NUMBER_OF_CLASSES, "errorLFCL", 0);
	initialize2D(gradientLFCL, NUMBER_OF_CLASSES, NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1, "gradientLFCL", 0);

	initialize1D(errorHL, NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1, "errorHL", 0);
	initialize2D(gradientHL, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, NUMBER_OF_INPUTS_HIDDEN_LAYER + 1, "gradientHL", 0);

	initialize1D(errorPL2, NUMBER_OF_INPUTS_HIDDEN_LAYER + 1, "errorPL2", 0);
	initialize3D(errorPL2_3D, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_POOLED_MAP_2, "errorPL2_3D", 0);

	initialize3D(errorCL2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2, "errorCL2", 0);
	initialize3D(kernelGradient_2, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, "kernelGradient_2", 0);

	initialize3D(errorPL1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_POOLED_MAP_1, "errorPL1", 0);

	initialize3D(errorCL1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_FEATURE_MAP_1, "errorCL1", 0);

	initialize3D(kernelGradient_1, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, "kernelGradient_1", 0);

	initialize2D(gradientLFCL_Acc, NUMBER_OF_CLASSES, NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1, "gradientLFCL_Acc", 0);
	initialize2D(gradientHL_Acc, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, NUMBER_OF_INPUTS_HIDDEN_LAYER + 1, "gradientHL_Acc", 0);
	initialize3D(kernelGradient_2_Acc, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, "kernelGradient_2_Acc", 0);
	initialize3D(kernelGradient_1_Acc, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, "kernelGradient_1_Acc", 0);

	initialize1D(biasOfConv_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, "biasOfConv_1", 1);
	initialize1D(biasOfConv_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, "biasOfConv_2", 1);

	initialize1D(biasOfConvGradAcc_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, "biasOfConvAcc_1", 0);
	initialize1D(biasOfConvGradAcc_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, "biasOfConvAcc_2", 0);

	openDataFile(x, y, "data/mnist_train.csv", NUMBER_OF_EXAMPLES);
        
	for(i = 0; i < NUMBER_OF_EXAMPLES; i++){

		imageInit(image, x[i], DIMENSIONS_OF_IMAGE, DIMENSIONS_OF_IMAGE);
		digitInit(digit, y[i], NUMBER_OF_CLASSES);

		convLayer(image, biasOfConv_1, kernel_1, sigmoid,
                    1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_FEATURE_MAP_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, featureMap_1);

		poolLayer(featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, maxPooledMap_1);

		convLayer(maxPooledMap_1, biasOfConv_2, kernel_2, sigmoid,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2, 
                    featureMap_2);

		poolLayer(featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, 
                    DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, maxPooledMap_2);
		transform3Dto1D(maxPooledMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
                    DIMENSIONS_OF_POOLED_MAP_2, DIMENSIONS_OF_POOLED_MAP_2, maxPooledMap_2_1D);

		fulConLayer(maxPooledMap_2_1D, biasHidden, weight1, sigmoid, NUMBER_OF_OUTPUTS_HIDDEN_LAYER,
                    NUMBER_OF_INPUTS_HIDDEN_LAYER, hiddenLayerOutput);

		fulConLayer(hiddenLayerOutput, biasLast, weight2, sigmoid, NUMBER_OF_CLASSES,
                    NUMBER_OF_OUTPUTS_HIDDEN_LAYER, output);

		//Update cost of the batch
		cost += costFunction(digit, output, NUMBER_OF_CLASSES);

		//Final Layer
		biasLastGradAcc += LFCLBackprop(digit, output, hiddenLayerOutput, biasLast,
                    NUMBER_OF_CLASSES, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, errorLFCL, gradientLFCL);

		//The only hidden layer
		biasHiddenGradAcc += FCLBackprop(errorLFCL, weight2, weight1, maxPooledMap_2_1D, biasHidden, sigmoidGradient,
                    NUMBER_OF_OUTPUTS_HIDDEN_LAYER, NUMBER_OF_CLASSES, NUMBER_OF_INPUTS_HIDDEN_LAYER,
                    errorHL, gradientHL);

		//Backpropagation through second pooling layer
		poolFromFCLBackprop(errorHL, weight1, NUMBER_OF_INPUTS_HIDDEN_LAYER,
                    NUMBER_OF_OUTPUTS_HIDDEN_LAYER, errorPL2);
		transform1Dto3D(errorPL2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
                    DIMENSIONS_OF_POOLED_MAP_2, DIMENSIONS_OF_POOLED_MAP_2, errorPL2_3D);

		//Backpropagation through second convolutional layer
		upSampleBackprop(errorPL2_3D, featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
                    DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, errorCL2);
		convLayerSigGrad(maxPooledMap_1, biasOfConv_2, kernel_2, sigmoidGradient,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, 
                    DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2, featureMap_2);
		elWiseMult3D(errorCL2, featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, 
                        DIMENSIONS_OF_FEATURE_MAP_2 ,DIMENSIONS_OF_FEATURE_MAP_2);
		convGrad(maxPooledMap_1, errorCL2,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2,
                    NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, kernelGradient_2);
		accumConvBiasGrad(errorCL2, biasOfConvGradAcc_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, 
                    DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2);

		//Backpropagation through first pooling layer
		convBackprop(errorCL2, kernel_2,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2,
                    NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_POOLED_MAP_1, DIMENSIONS_OF_POOLED_MAP_1, errorPL1);

		//Backpropagation through first convolutional layer
		upSampleBackprop(errorPL1, featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1,
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, errorCL1);
		convLayerSigGrad(image, biasOfConv_1, kernel_1, sigmoidGradient,
                    1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1,
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1, featureMap_1);
		elWiseMult3D(errorCL1, featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1);
		convGrad(image, errorCL1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1,
                    NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, 
                    kernelGradient_1);
		accumConvBiasGrad(errorCL1, biasOfConvGradAcc_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1);

		//Accumulate batch gradients
		accumGrad3D(kernelGradient_1_Acc, kernelGradient_1, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL);
		accumGrad3D(kernelGradient_2_Acc, kernelGradient_2, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL);
		accumGrad2D(gradientHL_Acc, gradientHL, NUMBER_OF_OUTPUTS_HIDDEN_LAYER, NUMBER_OF_INPUTS_HIDDEN_LAYER + 1);
		accumGrad2D(gradientLFCL_Acc, gradientLFCL, NUMBER_OF_CLASSES, NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1);

		//Update gradients at the end of the batch
		if((i + 1) % SIZE_OF_BATCH == 0){

			printf("Cost Function is %f at examples %d - %d\n", cost/SIZE_OF_BATCH, i - SIZE_OF_BATCH + 2, i + 1);
			cost = 0;

			updateGrad3D(kernel_1, kernelGradient_1_Acc, NUMBER_OF_KERNELS_CONV_LAYER_1,
					DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, LEARNING_RATE, SIZE_OF_BATCH);
			updateGrad3D(kernel_2, kernelGradient_2_Acc, NUMBER_OF_KERNELS_CONV_LAYER_2,
					DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL, LEARNING_RATE, SIZE_OF_BATCH);
			updateGrad2D(weight1, gradientHL_Acc, NUMBER_OF_OUTPUTS_HIDDEN_LAYER,
					NUMBER_OF_INPUTS_HIDDEN_LAYER + 1, LEARNING_RATE, SIZE_OF_BATCH);
			updateGrad2D(weight2, gradientLFCL_Acc, NUMBER_OF_CLASSES,
					NUMBER_OF_OUTPUTS_HIDDEN_LAYER + 1, LEARNING_RATE, SIZE_OF_BATCH);

			updateGrad1D(biasOfConv_1, biasOfConvGradAcc_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, BIAS_LEARNING_RATE, SIZE_OF_BATCH);
			updateGrad1D(biasOfConv_2, biasOfConvGradAcc_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, BIAS_LEARNING_RATE, SIZE_OF_BATCH);

			updateGrad(biasLast, biasLastGradAcc, BIAS_LEARNING_RATE, SIZE_OF_BATCH);
			updateGrad(biasHidden, biasHiddenGradAcc, BIAS_LEARNING_RATE, SIZE_OF_BATCH);
		}
	}

	printf("Testing:\n");
	for(i = 0; i < NUMBER_OF_EXAMPLES; i++) free(x[i]);
	openDataFile(x, y, "data/mnist_test.csv", NUMBER_OF_TEST_EXAMPLES);

	for(i = 0; i < NUMBER_OF_TEST_EXAMPLES; i++){

		imageInit(image, x[i], DIMENSIONS_OF_IMAGE, DIMENSIONS_OF_IMAGE);
		digitInit(digit, y[i], NUMBER_OF_CLASSES);

		convLayer(image, biasOfConv_1, kernel_1, sigmoid,
                    1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_FEATURE_MAP_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, featureMap_1);

		poolLayer(featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, 
                    DIMENSIONS_OF_FEATURE_MAP_1, DIMENSIONS_OF_FEATURE_MAP_1,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, maxPooledMap_1);

		convLayer(maxPooledMap_1, biasOfConv_2, kernel_2, sigmoid,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL, DIMENSIONS_OF_KERNEL,
                    NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2, 
                    featureMap_2);

		poolLayer(featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, 
                    DIMENSIONS_OF_FEATURE_MAP_2, DIMENSIONS_OF_FEATURE_MAP_2,
                    DIMENSIONS_OF_POOLING, DIMENSIONS_OF_POOLING, maxPooledMap_2);
		transform3Dto1D(maxPooledMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2,
                    DIMENSIONS_OF_POOLED_MAP_2, DIMENSIONS_OF_POOLED_MAP_2, maxPooledMap_2_1D);

		fulConLayer(maxPooledMap_2_1D, biasHidden, weight1, sigmoid, NUMBER_OF_OUTPUTS_HIDDEN_LAYER,
                    NUMBER_OF_INPUTS_HIDDEN_LAYER, hiddenLayerOutput);

		fulConLayer(hiddenLayerOutput, biasLast, weight2, sigmoid, NUMBER_OF_CLASSES,
                    NUMBER_OF_OUTPUTS_HIDDEN_LAYER, output);

		val = output[0];
		result = 0;
		for(j = 1; j < NUMBER_OF_CLASSES; j++){
			if(val < output[j]){
				val = output[j];
				result = j;
			}
		}
		printf("%d %d compared to %d\n", i, result, y[i]);

		if(result == y[i]){
			success++;
		}
	}
	printf("%4.2f%% of test cases were correct\n", success / (float) NUMBER_OF_TEST_EXAMPLES * 100);

	free3D(&image, 1, 28);

	free3D(&kernel_1, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL);
	free3D(&kernel_2, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL);

	free3D(&featureMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_FEATURE_MAP_1);
	free3D(&featureMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2);
	free3D(&maxPooledMap_1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_POOLED_MAP_1);
	free3D(&maxPooledMap_2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_POOLED_MAP_2);
	free(maxPooledMap_2_1D);
	free(hiddenLayerOutput);
	free(output);

	free2D(&weight1, NUMBER_OF_OUTPUTS_HIDDEN_LAYER);
	free2D(&weight2, NUMBER_OF_CLASSES);

	free(digit);

	free(errorLFCL);
	free2D(&gradientLFCL, NUMBER_OF_CLASSES);

	free(errorHL);
	free2D(&gradientHL, NUMBER_OF_OUTPUTS_HIDDEN_LAYER);

	free(errorPL2);
	free3D(&errorPL2_3D, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_POOLED_MAP_2);

	free3D(&errorCL2, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_2, DIMENSIONS_OF_FEATURE_MAP_2);
	free3D(&kernelGradient_2, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL);

	free3D(&errorPL1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_POOLED_MAP_1);

	free3D(&errorCL1, NUMBER_OF_FEATURE_MAPS_CONV_LAYER_1, DIMENSIONS_OF_FEATURE_MAP_1);

	free3D(&kernelGradient_1, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL);

	free2D(&gradientLFCL_Acc, NUMBER_OF_CLASSES);
	free2D(&gradientHL_Acc, NUMBER_OF_OUTPUTS_HIDDEN_LAYER);
	free3D(&kernelGradient_2_Acc, NUMBER_OF_KERNELS_CONV_LAYER_2, DIMENSIONS_OF_KERNEL);
	free3D(&kernelGradient_1_Acc, NUMBER_OF_KERNELS_CONV_LAYER_1, DIMENSIONS_OF_KERNEL);

	free(biasOfConv_1);
	free(biasOfConv_2);

	free(biasOfConvGradAcc_1);
	free(biasOfConvGradAcc_2);



	free2D(&x, NUMBER_OF_TEST_EXAMPLES);
	free(y);

	return 0;
}