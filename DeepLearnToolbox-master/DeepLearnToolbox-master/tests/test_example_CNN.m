%%SRCNN

close all;clear;clc;
addpath('/home/susu/Zhenghang/DeepLearnToolbox-master/DeepLearnToolbox-master/DeepLearnToolbox-master/data');
addpath('/home/susu/Zhenghang/DeepLearnToolbox-master/DeepLearnToolbox-master/DeepLearnToolbox-master/util');
addpath('/home/susu/Zhenghang/DeepLearnToolbox-master/DeepLearnToolbox-master/DeepLearnToolbox-master/CNN');

load train_test_data_2
Y = double(Y(:,:,1:128))/255;    % LowResulution images
X = double(X(:,:,1:128))/255;    % HighResolution images

%The setup of the SRCNN network
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer1
    struct('type', 'c', 'outputmaps', 32, 'kernelsize', 1) %convolution layer2
    struct('type', 'o', 'outputmaps', 1, 'kernelsize', 5) %convolution layer3
};


opts.alpha1 = 1e-4;% learning rate of the first two layers
opts.alpha2 = 1e-5;% learning rate of the last layer
opts.batchsize = 128; 
opts.numepochs = 1;
tic
% initialize the parameters for the SRCNN network
cnn = cnnsetup(cnn, Y, X);

cnn = cnntrain(cnn, Y, X, opts);

% [er, bad] = cnntest(cnn, test_x, test_y);
% 
% %plot mean squared error
plot(cnn.rL);
% %assert(er<0.12, 'Too big error');
% disp([num2str(er*100) '%error']);
toc
