function [ outputs, weightedSums ] = multiLayerPredict(inputs,weights,ONLFunc,HNLFunc)
% number of rows in weights corresponds to number of inputs
% number of columns = number of outputs
% NLFunc - nonlinear function used in neural net

layerInputs = input;
weightedSums = cell(length(weights));
for layer=1:length(weights)
    layerWeights = weights{layer};
    layerOutputs = zeros(size(layerWeights,2),1);
    layerWeightedSums = zeros(size(layerWeights,2),1);    
    for j=1:length(layerOutputs)
        layerWeightedSums(j) = layerWeights(1,j);
        for i=2:length(inputs)+1
            layerWeightedSums(j) = layerWeightedSums(j) + layerInputs(i-1)*layerWeights(i,j);
        end
        if i ~= length(weights)
            layerOutputs(j) = HNLFunc(weightedSums(j));
        else
            layerOutputs(j) = ONLFunc(weightedSums(j));
        end
    end
    weightedSums{layer} = weightedSums;
    layerInputs = layerOutputs;
end
outputs = layerOutputs;
end

