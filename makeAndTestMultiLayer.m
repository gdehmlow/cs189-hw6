function [trainingErrors,testErrors,trainingLosses,testLosses,epochList] = makeAndTestMultiLayer(data,labels,...
    ONLFunc,ONLDerivative,lossFunc,lossDerivative,HNLFunc,HNLDerivative,...
    layerSizes,epochs,reportFreq,stepSizeFunc)
% layerSizes - a vector containing the number of nodes at each layer in the
% network
% recommended step size is ~(0.5/(1+i
% ONLFunc - nonlinear function used in output layer of neural net e.g. @sigmoid.m
% ONLDerivative - the derivative of ONLFunc
% HNLFunc - nonlinear function used in hidden layers of neural net e.g.
% @(x)(1-tanh(x).^2)
% HNLDerivative - the derivative of HNLFunc
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
numLayers = length(layerSizes);
weights = cell(numLayers,1);
weights{1} = (rand(size(data,2)+1,layerSizes(1))*2-1)*(1/sqrt(size(data,2)+1));%10^-3;
for i=2:numLayers
    weights{i} = (rand(layerSizes(i-1)+1,layerSizes(i))*2-1)*(1/sqrt(layerSizes(i-1)+1));%10^-3;
end
resultsFile = fopen('multiLayerResults.txt','w');
epoch = 0;
bestTestError = inf;
bestTestWeights = 0;
trainingErrors = [];
trainingLosses = [];
testErrors = [];
testLosses = [];
epochList = [];
index = 1;
while epoch<epochs
    newWeights = trainMultiLayer(trainData,trainLabels,weights,ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,...
        lossFunc,lossDerivative,reportFreq,min(reportFreq,epochs-epoch),stepSizeFunc);
    weights = newWeights;
    [testError,testLoss] = testMultiLayer(testData,testLabels,weights,ONLFunc,HNLFunc,lossFunc);
    [trainError,trainLoss] = testMultiLayer(trainData,trainLabels,weights,ONLFunc,HNLFunc,lossFunc);
    epoch = epoch + reportFreq;
    if testError < bestTestError
         bestTestError = testError;
         bestTestWeights = weights;
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

