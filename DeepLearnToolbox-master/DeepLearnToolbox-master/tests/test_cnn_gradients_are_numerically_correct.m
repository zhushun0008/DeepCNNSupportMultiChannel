close all;
clear;
clc;
addpath('../util');
addpath('../CNN');
batch_x = rand(32,32,3,2);
batch_y = rand(20,20,3,2);
cnn.layers = {
    struct('type', 'i', 'inputmaps', 2) %input layer
    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer1
    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 1) %convolution layer2
    struct('type', 'o', 'outputmaps', 2, 'kernelsize', 5) %convolution layer3
};
cnn = cnnsetup(cnn, batch_x, batch_y);

cnn = cnnff(cnn, batch_x);
cnn = cnnbp(cnn, batch_y);
cnnnumgradcheck(cnn, batch_x, batch_y);