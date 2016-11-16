#include "mnist_classifier.hpp"

/* convLayer() performs the "valid" convolution, adds the bias and applies non-linearity.

   @param  input - Input to the convolutional layer.
   @param  bias - The bias unit for each feature map.
   @param  kernel - Kernels/filters.
   @param  nonLinFunc - function that is applied for non-linearity.
   @param  numberOfInputs - Number of inputs OR number of "float**" stored in parameter "input".
   @param  kernelRows - Height of the kernel.
   @param  kernelColumns - Width of the kernel.
   @param  numberOfFeatureMaps - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  featureMapRows - Height of the feature map.
   @param  featureMapColumns - Width of the feature map.
   @param  featureMap - Feature maps calculated this convolution OR output of this convolutional layer.
 */
void convLayer(float*** input, float* bias, float*** kernel, float (*nonLinFunc)(float z),
        int numberOfInputs, int kernelRows, int kernelColumns,
        int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
        float*** featureMap) {
    int i = 0, j = 0, k = 0, l = 0;
    int p = 0, q = 0;

    //Loops through every feature map/output of this convolutional layer
    for (i = 0; i < numberOfFeatureMaps; i++) {
        //Loops through the rows of the feature map
        for (j = 0; j < featureMapRows; j++) {
            //Loops through the columns of the feature map
            for (k = 0; k < featureMapColumns; k++) {
                //Sets feature map equal to zero
                featureMap[i][j][k] = 0;
                //Loops over every input to this convolutional layer
                for (l = 0; l < numberOfInputs; l++) {
                    //Loops through the rows of the kernel
                    for (p = 0; p < kernelRows; p++) {
                        //Loops through the columns of the kernel
                        for (q = 0; q < kernelColumns; q++) {
                            /* If there are 5 inputs and 50 feature maps, there will be 250 kernels,
                             * connecting every input to every feature map.
                             * "l * numberofFeatureMaps + i" causes kernels connected to the same input to be
                             * close to each other. In this particular example, this means the following:
                             * Kernels connected to the input[0] are in positions ranging from 0 to 49,
                             * kernels connected to the input[1] are in positions ranging from 50 to 99,
                             * kernels connected to the input[2] are in positions ranging from 100 to 149 and so on.
                             * This is done for the following reason.
                             * If you access the position of any kernel, let's call it "kernelPos",
                             * then "kernelPos / numberOfFeatureMaps" will give you the input it's connected to.
                             * And "kernelPos % numberOfFeatureMaps" will give you the feature map it's connected to.
                             *
                             * This is useful during the backpropagation.
                             */
                            featureMap[i][j][k] += input[l][j + p][k + q] * kernel[l * numberOfFeatureMaps + i][p][q];
                        }
                    }
                }
                //Adds bias unit to the feature map and applies the non-linearity.
                featureMap[i][j][k] = nonLinFunc(featureMap[i][j][k] + bias[i]);
            }
        }
    }
}

/* poolLayer() subsamples the input using max-pooling.
 *
 * @param  featureMap - Convoluted feature maps/inputs to the pooling layer.
   @param  numberOfFeatureMaps - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  featureMapRows - Height of the feature map.
   @param  featureMapColumns - Width of the feature map.
   @param  poolingRows - Factor by which height of the feature map is decreased.
   @param  poolingColumns - Factor by which width of the feature map is decreased.
   @param  maxPooledMap - Output of the max pooling layer.
 */
void poolLayer(float*** featureMap, int numberOfFeatureMaps,
        int featureMapRows, int featureMapColumns,
        int poolingRows, int poolingColumns, float*** maxPooledMap) {

    int i = 0, j = 0, k = 0;
    int p = 0, q = 0;

    //Variable that is used to find the maximum value of the region
    float max = 0;

    //Loops over every feature map/input
    for (i = 0; i < numberOfFeatureMaps; i++) {
        //Loops over the height coordinates that are in the origin of the pooling region
        for (j = 0; j < featureMapRows; j += poolingRows) {
            //Loops over the width coordinates that are in the origin of the pooling region
            for (k = 0; k < featureMapColumns; k += poolingColumns) {
                max = 0;
                //Loops through the height of the region
                for (p = 0; p < poolingRows; p++) {
                    //Loops through the width of the region
                    for (q = 0; q < poolingColumns; q++) {
                        //Looks for the maximum value of the region
                        if (max < featureMap[i][j + p][k + q]) {
                            max = featureMap[i][j + p][k + q];
                        }
                    }
                }
                //Sets the value of the output mapped to this region to the maximum value found in the region.
                maxPooledMap[i][j / poolingRows][k / poolingColumns] = max;
            }
        }
    }
}

/* @param  input - The activations from the previous layer or from the original image without the bias unit.
   @param  bias - Bias unit.
   @param  weight - The weights between the layers.
   @param  sizeOfInput - Number of units in the previous layer. DO NOT INCLUDE THE BIAS UNIT!
                                             It is accounted for inside the function.
   @param  sizeOfOutput - Number of units in this layer.
   @param  output - Activations from this layer.
 */
void fulConLayer(float* input, float bias, float** weight, float (*nonLinFunc)(float z),
        int sizeOfOutput, int sizeOfInput, float* output) {

    int i = 0, j = 0;

    //Loops through the outputs
    for (i = 0; i < sizeOfOutput; i++) {
        output[i] = 0;
        //Loops through all the inputs
        for (j = 0; j < sizeOfInput; j++) {
            output[i] += weight[i][j] * input[j];
        }
        /* Adds the bias unit(multiplied by its weight) and then applies non-linearity.
         * NOTE: The bias is used with the weight from the last column and not the first one.
         * It is important in backpropagation */
        output[i] = nonLinFunc(output[i] + weight[i][sizeOfInput] * bias);
    }
}

/* @param  trueValue - The label of the example in the vector form.
   @param  calculatedValue - The output of the final layer.
   @param  numberOfClasses - Number of units in the final layer.
   @return Cost of the example
 */
float costFunction(float* trueValue, float* calculatedValue, int numberOfClasses) {
    int i = 0;

    float cost = 0;

    /* Cross-entropy cost function is used
     * cost = - y*log(out) - (1-y)*log(1-out)
     *
     * positive = y*log(out)
     * negative = (1-y)*log(1-out)
     * From above, it should be clear why they are called positive and negative*/

    //The name is arbitrary and does not mean the number is positive
    float positive = 0;
    //The name is arbitrary and does not mean the number is negative
    float negative = 0;

    for (i = 0; i < numberOfClasses; i++) {
        positive += trueValue[i] * (logf(calculatedValue[i]));
    }
    for (i = 0; i < numberOfClasses; i++) {
        negative += (1 - trueValue[i]) * (logf((1 - calculatedValue[i])));
    }
    cost = -positive - negative;
    return cost;
}

/* LFCL stands for "Last fully connected layer".
 * It is used only for the output layer.

   @param  trueValue - The label of the example in the vector form.
   @param  calculatedValue - The output of the final layer.
   @param  input - Input to the last layer without the bias unit.
   @param  sizeOfLayer - Size of the final layer.
   @param  sizeOfNextLayer - Size of the next layer IN BACKPROPAGATION! DO NOT INCLUDE THE BIAS UNIT!
                                             It is accounted for inside the function.
   @param  error - Error OR the sensitivity list of the final layer.
   @param  gradient - Gradient for the weights between the last layer and the layer following it IN BACKPROPAGATION!
   @param  bias -  Bias unit.
   @return Bias Gradient of the bias unit of the next layer IN BACKPROPAGATION!

   "IN BACKPROPAGATION!" is emphasized because the order is now changed.
   Next layer in backpropagation is equivalent to the previous layer in feedforward and vice versa.
 */
float LFCLBackprop(float* trueValue, float* calculatedValue, float* input, float bias,
        int sizeOfLayer, int sizeOfNextLayer, float* error, float** gradient) {
    int i = 0, j = 0;
    float biasGrad = 0;

    //Computes the error.
    for (i = 0; i < sizeOfLayer; i++) {
        error[i] = calculatedValue[i] - trueValue[i];
    }

    //Computes the gradients of the weights
    for (i = 0; i < sizeOfLayer; i++) {
        for (j = 0; j < sizeOfNextLayer; j++) {
            gradient[i][j] = error[i] * input[j];
        }
        gradient[i][sizeOfNextLayer] = error[i] * bias;
    }

    //Computes the gradient of the bias
    for (i = 0; i < sizeOfLayer; i++) {
        biasGrad += error[i];
    }
    return biasGrad;
}

/* FCL stands for "Fully Connected Layer".

   @param  prevLayerError - The error OR sensitivity list from the previous layer in backpropagation.
   @param  prevTheta - Weights of the previous layer in backpropagation.
   @param  currentTheta - Weights of the current layer in backpropagation.
   @param  input - Input to the current layer without the bias unit.
   @param  bias -  Bias unit.
   @param  nonLinFuncDer - Derivative of the function that applied non-linearity during feedforward stage.
   @param  sizeOfLayer - Size of the current layer.
   @param  sizeOfPrevLayer - Size of the previous layer IN BACKPROPAGATION! Does not include the bias unit.
   @param  sizeOfNextLayer - Size of the next layer IN BACKPROPAGATION! Does not include the bias unit.
   @param  error - Error OR the sensitivity list of the final layer.
   @param  gradient - Gradient for the weights between the current layer and the layer following it IN BACKPROPAGATION!
   @return Bias gradient of for the bias unit of the next layer IN BACKPROPAGATION!

   "IN BACKPROPAGATION!" is emphasized because the order is now changed.
   Next layer in backpropagation is equivalent to the previous layer in feedforward and vice versa.
 */
float FCLBackprop(float* prevLayerError, float** prevTheta, float** currentTheta, float* input, float bias,
        float (*nonLinFuncDer)(float z), int sizeOfLayer, int sizeOfPrevLayer, int sizeOfNextLayer,
        float* error, float** gradient) {
    int i = 0, j = 0;
    float sigGrad[sizeOfLayer];
    float biasGrad = 0;

    /* Backpropagates the error from the previous layer.
     * NOTE: It was stated earlier, that the bias is used with the weight from the last column and not the first one.
     * Therefore, error[sizeOfLayer] is not going to be used for anything except calculating bias gradient*/
    for (i = 0; i < sizeOfLayer + 1; i++) {
        error[i] = 0;
        for (j = 0; j < sizeOfPrevLayer; j++) {
            error[i] += prevLayerError[j] * prevTheta[j][i];
        }
    }
    /*Computes feedforward again but this time applying the derivative of the non-linearity function*/
    for (i = 0; i < sizeOfLayer; i++) {
        sigGrad[i] = 0;
        for (j = 0; j < sizeOfNextLayer; j++) {
            sigGrad[i] += currentTheta[i][j] * input[j];
        }
        sigGrad[i] = nonLinFuncDer(sigGrad[i] + currentTheta[i][sizeOfNextLayer] * bias);

    }
    //Computes the error.
    for (i = 0; i < sizeOfLayer; i++) {
        error[i] = error[i] * sigGrad[i];
    }

    //Computes the graidents of the weights
    for (i = 0; i < sizeOfLayer; i++) {
        for (j = 0; j < sizeOfNextLayer; j++) {
            gradient[i][j] = error[i] * input[j];
        }
        gradient[i][sizeOfNextLayer] = error[i] * bias;
    }

    //Computes bias gradient
    for (i = 0; i < sizeOfLayer + 1; i++) {
        biasGrad += error[i];
    }

    return biasGrad;
}

/* poolFromFCLBackprop() is only used during backpropagation form the fully connected layer to the pooling layer.

   @param  prevLayerError - The error OR sensitivity list from the previous layer in backpropagation.
   @param  prevTheta - Weights of the previous layer in backpropagation.
   @param  sizeOfLayer - Size of the current layer.
   @param  sizeOfPrevLayer - Size of the previous layer IN BACKPROPAGATION! Does not include the bias unit.
   @param  error - Error OR the sensitivity list of the final layer.

   "IN BACKPROPAGATION!" is emphasized because the order is now changed.
   Next layer in backpropagation is equivalent to the previous layer in feedforward and vice versa.

 */
void poolFromFCLBackprop(float* prevLayerError, float** prevTheta,
        int sizeOfLayer, int sizeOfPrevLayer, float* error) {

    int i = 0, j = 0;
    //Backpropagates the error form the previous layer
    for (i = 0; i < sizeOfLayer; i++) {
        error[i] = 0;
        for (j = 0; j < sizeOfPrevLayer; j++) {
            error[i] += prevLayerError[j] * prevTheta[j][i];
        }
    }
}

/* upSampleBackprop() find the locations of the maximum values of all the regions.
 * Then it puts the error from the pooling layer into the corresponding locations.
 *
 * @param  poolingError - Error from the pooling error.
   @param  featureMap - Convoluted feature maps/inputs to the pooling layer. Needed here to find the maximum value again.
   @param  numberOfFeatureMaps - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  featureMapRows - Height of the feature map.
   @param  featureMapColumns - Width of the feature map.
   @param  poolingRows - Factor by which height of the feature map is decreased.
   @param  poolingColumns - Factor by which width of the feature map is decreased.
   @param  convError - Error into the convolutional layer

   This function performs the opposite of pooling/downsampling
 */
void upSampleBackprop(float*** poolingError, float*** featureMap,
        int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
        int poolingRows, int poolingColumns, float*** convError) {
    int i = 0, j = 0, k = 0;
    int p = 0, q = 0;

    //Variable that is used to find the maximum value of the region.
    float max = 0;
    //Variable that is used to find the location of the maximum value
    int pMax = 0, qMax = 0;

    for (i = 0; i < numberOfFeatureMaps; i++) {
        for (j = 0; j < featureMapRows; j += poolingRows) {
            for (k = 0; k < featureMapColumns; k += poolingColumns) {
                max = 0;
                for (p = 0; p < poolingRows; p++) {
                    for (q = 0; q < poolingColumns; q++) {
                        /* Sets all the errors to the convolutional layer equal to zero*/
                        convError[i][j + p][k + q] = 0;
                        //Finds the maximum value and its location
                        if (max < featureMap[i][j + p][k + q]) {
                            max = featureMap[i][j + p][k + q];
                            pMax = p;
                            qMax = q;
                        }
                    }
                }
                //Transfers the error to the cell where the maximum value is located
                convError[i][j + pMax][k + qMax] = poolingError[i][j / poolingRows][k / poolingColumns];
            }
        }
    }
}

/* convLayerSigGrad() does exactly the same thing as convLayer().
   The difference is that derivative of the non-linearity function is applied here, instead of the function itself.

   @param  input - Input to the convolutional layer.
   @param  bias - The bias unit for each feature map.
   @param  kernel - Kernels/filters.
   @param  nonLinFuncDer - Derivative of the function that was applied for non-linearity.
   @param  numberOfInputs - Number of inputs OR number of "float**" stored in parameter "input".
   @param  kernelRows - Height of the kernel.
   @param  kernelColumns - Width of the kernel.
   @param  numberOfFeatureMaps - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  featureMapRows - Height of the feature map.
   @param  featureMapColumns - Width of the feature map.
   @param  featureMap - Feature maps calculated this convolution OR output of this convolutional layer.
 */
void convLayerSigGrad(float*** input, float* bias, float*** kernel, float (*nonLinFuncDer)(float z),
        int numberOfInputs, int kernelRows, int kernelColumns,
        int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
        float*** featureMap) {
    int i = 0, j = 0, k = 0, l = 0;
    int p = 0, q = 0;

    for (i = 0; i < numberOfFeatureMaps; i++) {
        for (j = 0; j < featureMapRows; j++) {
            for (k = 0; k < featureMapColumns; k++) {
                featureMap[i][j][k] = 0;
                for (l = 0; l < numberOfInputs; l++) {
                    for (p = 0; p < kernelRows; p++) {
                        for (q = 0; q < kernelColumns; q++) {
                            featureMap[i][j][k] += input[l][j + p][k + q] * kernel[l * numberOfFeatureMaps + i][p][q];
                        }
                    }
                }
                featureMap[i][j][k] = nonLinFuncDer(featureMap[i][j][k] + bias[i]);
            }
        }
    }
}

/* Element-wise multiplication.
 * var1 must have the same size as var2
 * The result of var1*var2 is stored in var1 in order to save memory,
 * because the additional variable to store the result is not required for our purposes.
 */
void elWiseMult3D(float*** var1, float*** var2, int numberOfFeatureMaps, int featureMapRows, int featureMapColumns) {
    int i = 0, j = 0, k = 0;
    for (i = 0; i < numberOfFeatureMaps; i++) {
        for (j = 0; j < featureMapRows; j++) {
            for (k = 0; k < featureMapColumns; k++) {
                var1[i][j][k] *= var2[i][j][k];

            }
        }
    }
}

/* convGrad() performs "valid" convolution to find the gradients.
 *
 * @param  input - Input to the convolutional layer.
   @param  erroMap - The error of this convolutional layer.
   @param  numberOfFeatureMaps - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  featureMapRows - Height of the feature map.
   @param  featureMapColumns - Width of the feature map.
   @param  numberOfKernels - Number of kernels OR number of "float**" stored in parameter "kernel".
   @param  kernelRows - Height of the kernel.
   @param  kernelColumns - Width of the kernel.
   @param  kernelGradient - The gradients of the kernels.
 */
void convGrad(float*** input, float*** errorMap,
        int numberOfFeatureMaps, int featureMapRows, int featureMapColumns,
        int numberOfKernels, int kernelRows, int kernelColumns, float*** kernelGradient) {
    int i = 0, j = 0, k = 0;
    int p = 0, q = 0;

    //Loops over every kernel
    for (i = 0; i < numberOfKernels; i++) {
        //Loops over rows of the kernel
        for (j = 0; j < kernelRows; j++) {
            //Loops over columns of the kernel
            for (k = 0; k < kernelColumns; k++) {
                //Loops over rows of the feature map
                for (p = 0; p < featureMapRows; p++) {
                    //Loops over columns of the feature map
                    for (q = 0; q < featureMapColumns; q++) {
                        /* As it was explained in convLayer() kernels connect inputs to the feature maps in such a way,
                         * that "kernelPos / numberOfFeatureMaps" gives you the position of the input connected to it,
                         * and "kernelPos % numberOfFeatureMaps" gives you the position of the feature map connected to it */
                        kernelGradient[i][j][k] +=
                                input[i / numberOfFeatureMaps][j + p][k + q] * errorMap[i % numberOfFeatureMaps][p][q];
                    }
                }
            }
        }
    }
}

//Accumulates the bias gradient of the convolutional layer
void accumConvBiasGrad(float*** error, float* biasGradAcc, int numberOfFeatureMaps, int featureMapRows, int featureMapColumns) {
    int i = 0, j = 0, k = 0;

    for (i = 0; i < numberOfFeatureMaps; i++) {
        for (j = 0; j < featureMapRows; j++) {
            for (k = 0; k < featureMapColumns; k++) {
                biasGradAcc[i] += error[i][j][k];
            }
        }
    }
}

/* convBackprop() propagates the error from the convolutional layer to the pooling layer.
 * It performs "full" convolution.
 *
 * @param  errorIn - Error from the convolutional layer.
   @param  kernel - The kernel that was used during the feedforward stage.
   @param  numberOfErrorIns - Number of feature maps OR number of "float**" stored in parameter "featureMap".
   @param  errorInRows - Height of the feature map.
   @param  errorInColumns - Width of the feature map.
   @param  numberOfKernels - Number of kernels OR number of "float**" stored in parameter "kernel".
   @param  kernelRows - Height of the kernel.
   @param  kernelColumns - Width of the kernel.
   @param  numberOfErrorOuts - Number of kernels OR number of "float**" stored in parameter "kernel".
   @param  errorOutRows - Height of the kernel.
   @param  errorOutColumns - Width of the kernel.
   @param  errorOut - Error into the pooling Layer.
 */
void convBackprop(float*** errorIn, float*** kernel,
        int numberOfErrorIns, int errorInRows, int errorInColumns,
        int numberOfKernels, int kernelRows, int kernelColumns,
        int numberOfErrorOuts, int errorOutRows, int errorOutColumns, float*** errorOut) {
    int i = 0, j = 0, k = 0;
    int p = 0, q = 0;

    for (i = 0; i < numberOfErrorOuts; i++) {
        for (j = 0; j < errorOutRows; j++) {
            for (k = 0; k < errorOutColumns; k++) {
                errorOut[i][j][k] = 0;
            }
        }
    }

    for (i = 0; i < numberOfKernels; i++) {
        for (j = 0; j < errorInRows; j++) {
            for (k = 0; k < errorInColumns; k++) {
                for (p = 0; p < kernelRows; p++) {
                    for (q = 0; q < kernelColumns; q++) {
                        /* ErrorOut is going to the input
                         * ErrorIn is coming from the feature map
                         * Therefore, referring to convLayer() where,
                         * kernelpos / numberOfFeatureMaps = inputPos,
                         * kernelpos % numberOfFeatureMaps = featureMapPos.
                         * In this case it means:
                         * kernelpos / numberOfErrorIns = errorOutPos,
                         * kernelpos % numberOfErrorIns = errorInPos.*/
                        errorOut[i / numberOfErrorIns][j + p][k + q] +=
                                errorIn[i % numberOfErrorIns][j][k] * kernel[i][p][q];
                    }
                }
            }
        }
    }
}

//Updates the weights with the gradients accumulated during the mini-batch
void updateGrad3D(float*** weight, float*** gradientAcc, int numberOfElements,
        int numberOfRows, int numberOfColumns, float learningRate, int sizeOfBatch) {
    int i = 0, j = 0, k = 0;

    for (i = 0; i < numberOfElements; i++) {
        for (j = 0; j < numberOfRows; j++) {
            for (k = 0; k < numberOfColumns; k++) {
                weight[i][j][k] = weight[i][j][k] - learningRate * gradientAcc[i][j][k] / sizeOfBatch;
                gradientAcc[i][j][k] = 0;
            }
        }
    }
}

//Accumulates the gradients for the duration of the mini-batch
void accumGrad3D(float*** gradientAcc, float*** gradient, int numberOfElements, int numberOfRows, int numberOfColumns) {
    int i = 0, j = 0, k = 0;

    for (i = 0; i < numberOfElements; i++) {
        for (j = 0; j < numberOfRows; j++) {
            for (k = 0; k < numberOfColumns; k++) {
                gradientAcc[i][j][k] = gradientAcc[i][j][k] + gradient[i][j][k];
                gradient[i][j][k] = 0;
            }
        }
    }
}

//Updates the weights with the gradients accumulated during the mini-batch
void updateGrad2D(float** weight, float** gradientAcc, int numberOfRows, int numberOfColumns, float learningRate,
        int sizeOfBatch) {
    int i = 0, j = 0;

    for (i = 0; i < numberOfRows; i++) {
        for (j = 0; j < numberOfColumns; j++) {
            weight[i][j] = weight[i][j] - learningRate * gradientAcc[i][j] / sizeOfBatch;
            ;
            gradientAcc[i][j] = 0;
        }
    }
}

//Accumulates the gradients for the duration of the mini-batch
void accumGrad2D(float** gradientAcc, float** gradient, int numberOfRows, int numberOfColumns) {
    int i = 0, j = 0;

    for (i = 0; i < numberOfRows; i++) {
        for (j = 0; j < numberOfColumns; j++) {
            gradientAcc[i][j] = gradientAcc[i][j] + gradient[i][j];
            gradient[i][j] = 0;
        }
    }
}

//Updates the weights with the gradients accumulated during the mini-batch
void updateGrad1D(float* toUpdate, float* gradientAcc, int numberOfElements, float learningRate, int sizeOfBatch) {
    int i = 0;

    for (i = 0; i < numberOfElements; i++) {
        toUpdate[i] = toUpdate[i] - learningRate * gradientAcc[i] / sizeOfBatch;
        gradientAcc[i] = 0;
    }
}

//Updates the weights with the gradients accumulated during the mini-batch
void updateGrad(float toUpdate, float gradientAcc, float learningRate, int sizeOfBatch) {

    toUpdate = toUpdate - learningRate * gradientAcc / sizeOfBatch;
    gradientAcc = 0;
}

//Non-linearity function.  
float sigmoid(float z) {
    return ((1 / (1 + expf(-z))));
}

//Derivative of the non-linearity function
float sigmoidGradient(float z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

//Transforms y from scalar into a vector
void digitInit(float* digit, int y, int numberOfClasses) {

    int i = 0;
    for (i = 0; i < numberOfClasses; i++) {
        digit[i] = 0;
    }
    digit[y] = 1;
}

//Initializes the image
void imageInit(float*** image, float* x, int imageRows, int imageColumns) {
    int i = 0, j = 0;
    for (i = 0; i < imageRows; i++) {
        for (j = 0; j < imageColumns; j++) {
            image[0][i][j] = x[i * imageColumns + j];
        }
    }
}

//Identifies the digit based on the output of the final layer
float computeResult(float* output) {

    int j = 0;
    float val = 0;
    float result = 0;

    //Assume the digit is 0
    val = output[0];
    result = 0;

    //Change the digit to the location of the highest output
    for (j = 1; j < NUMBER_OF_CLASSES; j++) {
        if (val < output[j]) {
            val = output[j];
            result = j;
        }
    }
    return result;
}
