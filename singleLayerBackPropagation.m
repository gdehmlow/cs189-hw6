function [ weightGradients, biasGradients ] = singleLayerBackPropagation(points,labels,weights,biases,NLFunc,NLDerivative,lossDerivative)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

weightGradients = zeros(size(weights));
biasGradients = zeros(size(biases));
for p=1:length(labels)
    labelVector  = zeros(size(weights,2));
    labelVector(labels(p)+1) = 1;
    [outputs,weightedSums] = singleLayerPredict(points(p,:),weights,biases,NLFunc);
    for j=1:length(outputs)
        lossD = lossDerivative(outputs(j),labelVector(j));
        NLD = NLDerivative(weightedSums(j));
        biasGradients(j) = lossD*NLD;
        for i=1:size(points,2)
            weightGradients(i,j) = weightGradients(i,j) + lossD*NLD*points(p,i);
        end
    end
end
weightGradients = weightGradients/4;
biasGradients = biasGradients/4;
end

