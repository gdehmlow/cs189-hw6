function [ outputs, weightedSums ] = singleLayerPredict(inputs,weights,NLFunc)
% number of rows in weights corresponds to number of inputs
% number of columns = number of outputs
% NLFunc - nonlinear function used in neural net
if (length(inputs)+1 ~= size(weights,1))
    fprintf('input size must be one less than number of rows of weights');
end
outputs = zeros(size(weights,2),1);
weightedSums = zeros(size(weights,2),1);
for j=1:length(outputs)
    weightedSums(j) = weights(1,j);
    %weightedSums(j) = biases(j);
    for i=2:length(inputs)+1
        weightedSums(j) = weightedSums(j) + inputs(i-1)*weights(i,j);
    end
    outputs(j) = NLFunc(weightedSums(j));
end

end

