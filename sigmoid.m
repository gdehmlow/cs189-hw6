function [ value ] = sigmoid(weightedSumPlusBias)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

value = 1./(1+exp(-(weightedSumPlusBias)));

end

