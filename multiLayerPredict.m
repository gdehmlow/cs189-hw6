function [ outputs, weightedSums ] = multiLayerPredict(inputs,weights,ONLFunc,HNLFunc)
% number of rows in weights corresponds to number of inputs
% number of columns = number of outputs
% NLFunc - nonlinear function used in neural net

layerInputs = [1;inputs'];
weightedSums = cell(length(weights),1);
outputs = cell(length(weights),1);
for layer=1:length(weights)
    layerWeights = weights{layer};
    layerWeightedSums = layerWeights'*layerInputs;
    %for j=1:length(layerOutputs)
    %    layerWeightedSums(j) = layerWeights(1,j);
     %   for i=2:length(layerInputs)+1
      %      layerWeightedSums(j) = layerWeightedSums(j) + layerInputs(i-1)*layerWeights(i,j);
       % end
    %end
    if layer ~= length(weights)
            func = HNLFunc;
    else
            func = ONLFunc;
    end
    layerOutputs = func(layerWeightedSums);
    weightedSums{layer} = layerWeightedSums;
    outputs{layer} = layerOutputs;
    layerInputs = [1;layerOutputs];
end
end

