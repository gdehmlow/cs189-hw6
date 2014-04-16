function [ outputs, weightedSums ] = singleLayerPredict(inputs,weights,biases,NLFunc)
% number of rows in weights corresponds to number of inputs
% number of columns = number of outputs
% NLFunc - nonlinear function used in neural net
if (length(inputs) ~= size(weights,1))
    fprintf('input size must match number of rows of weights');
end
outputs = zeros(size(weights,2),1);
weightedSums = zeros(size(weights,2),1);
for j=1:length(outputs)
    weightedSums(j) = biases(j);
    for i=1:length(inputs)
        weightedSums(j) = weightedSums(j) + inputs(i)*weights(i,j);
    end
    outputs(j) = NLFunc(weightedSums(j));
end

end

