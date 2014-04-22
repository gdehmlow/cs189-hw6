function [ bestTestError, bestTestWeights, bestEpoch] = makeAndTestMultiLayer(data,labels,...
    numOutputs,ONLFunc,ONLDerivative,...
    numLayers,layerSizes,lossFunc,lossDerivative,HNLFunc,HNLDerivative,...
    epochs,reportFreq,stepSizeFunc)
% NLFunc - nonlinear function used in neural net e.g. @sigmoid.m
% NLDerivative - the derivative of NLFunc
% lossFunc - the loss function e.g. @meanSquaredLoss.m or @crossEntropyLoss.m
% lossDerivative - the derivative of lossFunc

if size(data,3) ~= 1
    newData = zeros(size(data,3),size(data,1)*size(data,2));
    for i=1:size(data,3)
        imageVector = reshape(data(:,:,i),1,size(data,1)*size(data,2));
        newData(i,:) = double(imageVector)/norm(double(imageVector));
    end 
end

data = newData;
% organize data, pseudorandomly divide it into sets
s = RandStream('mt19937ar','Seed',0);
randIndexes = randperm(s,size(data,1));
testData = data(randIndexes(1:floor(size(data,1)/10)),:);
testLabels = labels(randIndexes(1:floor(size(data,1)/10)));
trainData = data(randIndexes(floor(size(data,1)/10)+1:size(data,1)),:);
trainLabels = labels(randIndexes(floor(size(data,1)/10)+1:size(data,1)));

%instantiate weights and biases, train them
numLayers = numLayers + 1;
layerSizes = [layerSizes,numOutputs];
weights = cell(numlayers,1);
for i=1:numLayers-1
    weights{i} = (rand(layerSizes(i+1)+1,layerSizes(i))*2-1)*10^-3;
weights{numLayers} = (rand(size(data,2)+1,layerSizes(numLayers))*2-1)*10^-3;
resultsFile = fopen('results.txt','w');
epoch = 0;
bestTestError = inf;
bestTestWeights = 0;
while epoch<epochs
    weights = trainMultiLayer(trainData,trainLabels,weights,ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,...
        lossDerivative,reportFreq,min(reportFreq,epochs-epoch),stepSizeFunc);
    [testError,testLoss] = testMultiLayer(testData,testLabels,numOutputs,weights,NLFunc,lossFunc);
    [trainError,trainLoss] = testMultiLayer(trainData,trainLabels,numOutputs,weights,NLFunc,lossFunc);
    epoch = epoch + reportFreq;
    if testError < bestTestError
         bestTestError = testError;
         bestTestWeights = weights;
         bestEpoch = epoch;
    end
    fprintf(resultsFile,'\nEpoch: %d',epoch);
    fprintf(resultsFile,'\n\ttrainingError:\t');
    fprintf(resultsFile,'%f', trainError);
    fprintf(resultsFile,'\n\ttrainingLoss:\t');
    fprintf(resultsFile,'%f', trainLoss);
    fprintf(resultsFile,'\n\ttestError:\t');
    fprintf(resultsFile, '%f', testError);
    fprintf(resultsFile,'\n\ttestLoss:\t');
    fprintf(resultsFile, '%f', testLoss);
end
fprintf(resultsFile,'\nBest Epoch: %d',bestEpoch);
fprintf(resultsFile,'\n\tbestTestError:\t');
fprintf(resultsFile,'%f', bestTestError);
end

