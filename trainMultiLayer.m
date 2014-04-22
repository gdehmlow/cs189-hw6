function [ weights ] = trainMultiLayer(data,labels,weights,ONLFunc,ONLDerivative,...
    HNLFunc,HNLDerivative,lossDerivative,epochs,startEpoch,stepSizeFunc)
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
        weightGradients = multiLayerBackPropagation(miniBatchData,miniBatchLabels,...
            weights,ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,lossDerivative);
        weights = weights - weightGradients*stepSizeFunc(startEpoch+i);
    end
end

end


