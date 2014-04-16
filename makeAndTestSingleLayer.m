function [ error, loss, weights, biases ] = makeAndTestSingleLayer(data,labels,numOutputs,NLFunc,...
    NLDerivative,lossFunc,lossDerivative,epochs,stepSizeFunc)
% NLFunc - nonlinear function used in neural net e.g. @sigmoid.m
% NLDerivative - the derivative of NLFunc
% lossFunc - the loss function e.g. @meanSquaredLoss.m or @crossEntropyLoss.m
% lossDerivative - the derivative of lossFunc
if size(data,3) ~= 1
    newData = zeros(size(data,3),size(data,1)*size(data,2));
    for i=1:size(data,3)
        newData(i,:) = reshape(data(:,:,i),1,size(data,1)*size(data,2));
    end
end 
data = newData/255;
s = RandStream('mt19937ar','Seed',0);
randIndexes = randperm(s,size(data,1));
testData = data(randIndexes(1:floor(size(data,1)/10)),:);
testLabels = labels(randIndexes(1:floor(size(data,1)/10)));
trainData = data(randIndexes(floor(size(data,1)/10)+1:size(data,1)),:);
trainLabels = labels(randIndexes(floor(size(data,1)/10)+1:size(data,1)));
[weights,biases] = trainSingleLayer(trainData,trainLabels,numOutputs,NLFunc,NLDerivative,...
    lossDerivative,epochs,stepSizeFunc);
[error,loss] = testSingleLayer(testData,testLabels,numOutputs,weights,biases,NLFunc,lossFunc);
end

