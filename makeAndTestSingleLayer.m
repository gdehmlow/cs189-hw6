function [trainingErrors,testErrors,trainingLosses,testLosses,epochList] = makeAndTestSingleLayer(data,labels,numOutputs,NLFunc,...
    NLDerivative,lossFunc,lossDerivative,epochs,reportFreq,stepSizeFunc)
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
weights = (rand(size(data,2)+1,numOutputs)*2-1)*10^-3;
%biases = (rand(numOutputs,1)*2-1)*10^-3;
resultsFile = fopen('singleLayerResults.txt','w');
epoch = 0;
bestTestError = inf;
bestTestWeights = 0;
%bestTestBiases = 0;
trainingErrors = [];
trainingLosses = [];
testErrors = [];
testLosses = [];
epochList = [];
index = 1;
while epoch<epochs
    weights = trainSingleLayer(trainData,trainLabels,weights,NLFunc,NLDerivative,...
        lossDerivative,reportFreq,min(reportFreq,epochs-epoch),stepSizeFunc);
    [testError,testLoss] = testSingleLayer(testData,testLabels,numOutputs,weights,NLFunc,lossFunc);
    [trainError,trainLoss] = testSingleLayer(trainData,trainLabels,numOutputs,weights,NLFunc,lossFunc);
    epoch = epoch + reportFreq;
    if testError < bestTestError
         bestTestError = testError;
         bestTestWeights = weights;
         %bestTestBiases = biases;
         bestEpoch = epoch;
    end
    trainingErrors(index) = trainError;
    trainingLosses(index) = trainLoss;
    testErrors(index) = testError;
    testLosses(index) = testLoss;
    epochList(index) = epoch;
    index = index + 1;
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

