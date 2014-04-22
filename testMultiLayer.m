function [ error, loss ] = testMultiLayer(testData,testLabels,weights,ONLFunc,HNLFunc,lossFunc)
% NLFunc - nonlinear function used in neural net
% lossFunc - the loss function

predictedLabels = zeros(size(testData,1),size(weights{length(weights)},2));
testLabelVectors = zeros(size(testData,1),size(weights{length(weights)},2));
errors = 0;
for p=1:size(testData,1)
    outputs = multiLayerPredict(testData(p,:),weights,ONLFunc,HNLFunc);
    predictedLabels(p,:) = outputs{length(outputs)};
    testLabelVectors(p,testLabels(p)+1)= 1;
    if find(predictedLabels(p,:)== max(predictedLabels(p,:))) ~= testLabels(p)+1
        errors = errors + 1;
    end
end
loss = lossFunc(predictedLabels,testLabelVectors);
error = errors/size(testData,1);
end

