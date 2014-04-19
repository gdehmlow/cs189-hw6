function [ weightGradients ] = singleLayerBackPropagation(points,labels,weights,NLFunc,NLDerivative,lossDerivative)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

weightGradients = zeros(size(weights));
%biasGradients = zeros(size(biases));
for p=1:length(labels)
    labelVector  = zeros(size(weights,2));
    labelVector(labels(p)+1) = 1;
    [outputs,weightedSums] = singleLayerPredict(points(p,:),weights,NLFunc);
    for j=1:length(outputs)
        lossD = lossDerivative(outputs(j),labelVector(j));
        NLD = NLDerivative(weightedSums(j));
        weightGradients(1,j) = lossD*NLD;
        for i=2:size(points,2)+1
            weightGradients(i,j) = weightGradients(i,j) + lossD*NLD*points(p,i-1);
        end
    end
end
end

