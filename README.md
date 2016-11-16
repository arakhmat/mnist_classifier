#mnist_classifier

Convolutional neural network for digit recognition implemented in C++.

## Description

The neural network has the structure of LeNet.

LeNet:
<p align="center">
<a href="http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/"><img alt="LeNet" src="http://www.pyimagesearch.com/wp-content/uploads/2016/06/lenet_architecture-768x226.png"/></a>
</p>

By default the hyperparameters are:

<<<<<<< HEAD
Input image:                1\*28\*28<br />
First convolutional layer:  5\*28\*28<br />
First max-pooling layer:    5\*24\*24<br />
Second convolutional layer: 20\*12\*12<br />
Second max-pooling layer:   20\*8\*8<br />
First hidden layer:         100<br />
Output layer:               10<br />
<br />
Batch size:                 50<br />
Number of epochs:           1<br />
Learning rate:              0.5<br />
Cost function:              Mean Squared Error<br />

The input data is normalized. There is no momentum or weight decay.<br />

The classifier is trained with 60000 examples and then it is tested with 10000 examples. With these hyperparameters, it achieves an accuracy of more than 94% depending on the initialization of the weights.<br /> 

## How to run

In the root directory type:<br />
	./download_datasets.sh<br />
	make && bin/mnist_classifier<br />
=======
Input image:                1*28*28
First convolutional layer:  5*28*28
First max-pooling layer:    5*24*24
Second convolutional layer: 20*12*12
Second max-pooling layer:   20*8*8
First hidden layer:         100
Output layer:               10

Batch size:                 50
Number of epochs:           1
Learning rate:              0.5
Cost function:              Mean Squared Error 

The input data is normalized. There is no momentum or weight decay.

The classifier is trained with 60000 examples and then it is tested with 10000 examples. With these hyperparameters, it achieves an accuracy of more than 94% depending on the initialization of the weights. 

## How to run

In the root directory type:
	./donwload_datasets.sh
	make && bin/mnist_classifier
>>>>>>> cbb9f2c9841d61ebac1539e44dffc7e8005c345e

## Possible Improvements

- Implement a better algorithm for backpropagation (currently gradient descent is used)
- Add functions that would store and load the weights
- Add momentum and weight decay
- Flatten 2D and 3D arrays

