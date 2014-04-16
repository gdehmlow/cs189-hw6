function [ weights,biases ] = trainSingleLayer(data,labels,numOutputs,NLFunc,...
    NLDerivative,lossDerivative,epochs,stepSizeFunc)
% NLFunc - nonlinear function used in neural net
% NLDerivative - the derivative of NLFunc
% lossFunc - the loss function
% lossDerivative - the derivative of lossFunc
weights = (rand(size(data,2),numOutputs)*2-1)*10^-3;
biases = (rand(numOutputs,1)*2-1)*10^-3;
for i=1:epochs
    randIndexes = randperm(size(data,1));
    data = data(randIndexes,:);
    labels = labels(randIndexes);
    chunkSize = max(1,floor(size(data,1)/200));
    for n=0:min(size(data,1),200)-1
        miniBatchData = data(chunkSize*n+1:min(chunkSize*(n+1),size(data,1)),:);
        miniBatchLabels = labels(chunkSize*n+1:min(chunkSize*(n+1),size(data,1)));
        [weightGradients,biasGradients] = singleLayerBackPropagation(miniBatchData,...
            miniBatchLabels,weights,biases,NLFunc,NLDerivative,lossDerivative);
        weights = weights - weightGradients*stepSizeFunc(i);
        biases = biases - biasGradients*stepSizeFunc(i);
    end
end

end


