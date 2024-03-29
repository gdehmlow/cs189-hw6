function [ weights ] = trainMultiLayer(data,labels,weights,ONLFunc,ONLDerivative,...
    HNLFunc,HNLDerivative,lossFunc,lossDerivative,epochs,startEpoch,stepSizeFunc)
% NLFunc - nonlinear function used in neural net
% NLDerivative - the derivative of NLFunc
% lossFunc - the loss function
% lossDerivative - the derivative of lossFunc

for i=1:epochs
    randIndexes = randperm(size(data,1));
    data = data(randIndexes,:);
    labels = labels(randIndexes);
    chunkSize = max(1,floor(size(data,1)/200));
    for n=0:min(size(data,1),200)-1
        miniBatchData = data(chunkSize*n+1:min(chunkSize*(n+1),size(data,1)),:);
        miniBatchLabels = labels(chunkSize*n+1:min(chunkSize*(n+1),size(data,1)));
        %correctWeightGradients = finiteDifferences(miniBatchData,miniBatchLabels,weights,ONLFunc,HNLFunc,lossFunc);
        weightGradients = multiLayerBackPropagation(miniBatchData,miniBatchLabels,weights,ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,lossDerivative);
        for layer=1:length(weights)
            weights{layer} = weights{layer} - weightGradients{layer}*stepSizeFunc(startEpoch+i);
        end
    end
end

end


