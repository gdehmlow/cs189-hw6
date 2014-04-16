function [ value ] = crossEntropyLossDerivative(output,label)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

value = output-label;
value = value/(output*(1-output));
end

