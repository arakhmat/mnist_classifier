# mnist_classifier

Convolutional neural network for digit recognition implemented in C.

## Description

The neural network has the structure of LeNet.

LeNet:
<p align="center">
<a href="http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/"><img alt="LeNet" src="http://www.pyimagesearch.com/wp-content/uploads/2016/06/lenet_architecture-768x226.png"/></a>
</p>

By default the hyperparameters are:

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
Cost function:              Cross-Entropy<br />

The input data is normalized. There is no momentum or weight decay.<br />

The classifier is trained with 60000 examples and then it is tested with 10000 examples. With these hyperparameters, it achieves an accuracy of more than 94% depending on the initialization of the weights.<br /> 

## Usage

To build :
```sh
make
```

To download the datasets:
```sh
./download_datasets.sh
```

To run:
```sh
bin/mnist_classifier
```

## Possible Improvements

- Implement a better algorithm for backpropagation (currently gradient descent is used)
- Add functions that would store and load the weights
- Add momentum and weight decay
- Flatten 2D and 3D arrays
- Add more activation functions

