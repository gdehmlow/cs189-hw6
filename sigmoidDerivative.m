function [ value ] = sigmoidDerivative(weightedSumPlusBias)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

value = exp(-weightedSumPlusBias);
value = value/((1+exp(-weightedSumPlusBias))^2);
end

