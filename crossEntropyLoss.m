function [ loss ] = crossEntropyLoss(predictedLabels,trueLabels)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
loss = 0;
for p=1:size(predictedLabels,1)
    for i=1:size(predictedLabels,2)
        loss = loss + trueLabels(p,i)*reallog(predictedLabels(p,i)+10^-6);
        loss = loss + (1-trueLabels(p,i))*reallog(1-predictedLabels(p,i)+10^-6);
    end
end
loss = -loss;
end

