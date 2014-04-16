function [ loss ] = meanSquaredLoss(predictedLabels,trueLabels)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
loss = 0;
for p=1:length(predictedLabels)
    lossVector = (trueLabels(p,:) - predictedLabels(p,:));
    loss = loss + lossVector*lossVector';
end
loss = loss/2;
end

