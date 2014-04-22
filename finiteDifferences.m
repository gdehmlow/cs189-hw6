function [ weightGradients ] = finiteDifferences(points,labels,weights,ONLFunc,HNLFunc,lossFunc)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

difference = 10^-4;
weightGradients = cell(length(weights),1);
for layer=1:length(weights)
    layerWeights = weights{layer};
    layerWeightGradients = zeros(size(weights{layer}));
    for i=1:size(layerWeights,1)
        for j=1:size(layerWeights,2)
            [originalError,originalLoss] = testMultiLayer(points,labels,weights,ONLFunc,HNLFunc,lossFunc);
            weights{layer}(i,j) = weights{layer}(i,j) + difference;
            [newError,newLoss] = testMultiLayer(points,labels,weights,ONLFunc,HNLFunc,lossFunc);
            weights{layer}(i,j) = weights{layer}(i,j) - difference;
            layerWeightGradients(i,j) = (newLoss - originalLoss)/difference;
        end
    end
    weightGradients{layer} = layerWeightGradients;
end

