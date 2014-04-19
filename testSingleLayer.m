function [ error, loss ] = testSingleLayer(testData,testLabels,numOutputs,weights,NLFunc,lossFunc)
% NLFunc - nonlinear function used in neural net
% lossFunc - the loss function
predictedLabels = zeros(size(testData,1),numOutputs);
testLabelVectors = zeros(size(testData,1),numOutputs);
errors = 0;
for p=1:size(testData,1)
    predictedLabels(p,:) = singleLayerPredict(testData(p,:),weights,NLFunc);
    testLabelVectors(p,testLabels(p)+1)= 1;
    if find(predictedLabels(p,:)== max(predictedLabels(p,:))) ~= testLabels(p)+1
        errors = errors + 1;
    end
end
loss = lossFunc(predictedLabels,testLabelVectors);
error = errors/size(testData,1);
end

