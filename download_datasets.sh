#!/bin/bash

mkdir -p data
wget http://pjreddie.com/media/files/mnist_train.csv -O data/mnist_train.csv
wget http://pjreddie.com/media/files/mnist_test.csv -O data/mnist_test.csv
echo "MNIST datasets were downloaded"