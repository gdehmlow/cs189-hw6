function [ value ] = crossEntropyLossDerivative(output,label)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

value = output-label;
value = value/((output+10^-7)*((1-output)+10^-7));
end

