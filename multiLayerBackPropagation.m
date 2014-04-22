function [ weightGradients ] = multiLayerBackPropagation(points,labels,weights,...
    ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,lossDerivative)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

weightGradients = cell(length(weights));
deltas = cell(length(weights));
%toplayer computation then propagation
for i=1:length(weights)
    weightGradients{i} = zeros(size(weights{i}));
    deltas{i} = zeros(size(weights{i},2),1);
end
for p=1:length(labels)
    labelVector  = zeros(length(labels));
    labelVector(labels(p)+1) = 1;
    [outputs, weightedSums] = multiLayerPredict(points(p,:),weights,ONLFunc,HNLFunc);
    for j=1:length(outputs)
        lossD = lossDerivative(outputs(j),labelVector(j));
        ONLD = ONLDerivative(weightedSums{length(weights)}(j));
        deltas{length(weights)}(j) = lossD*ONLD;
        weightGradients{length(weights)}(1,j) = weightGradients{length(weights)}(i,j) + deltas{length(weights)}(j);
        for i=2:size(points,2)+1
            weightGradients{length(weights)}(i,j) = weightGradients{length(weights)}(i,j) + deltas{length(weights)}(j)*points(p,i-1);
        end
    end
    for layer=1:length(weights)-1
        for i=1:size(weights{length(weights)-layer},1)
            for j=1:size(weights{length(weights)-layer+1},1)
                deltas{length(weights)-layer}(i) = deltas{length(weights)-layer}(i) + ...
                    weights{length(weights)-layer+1}(i,j)*deltas{length(weights)-layer+1}(j);
            end
        end
        for j=1:size(weights{length(weights)-layer},1)
            HNLD = HNLDerivative(weightedSums{length(weights)-layer}(j));
            deltas{length(weights)-layer}(j) = deltas{length(weights)-layer}(j)*HNLD;
            weightGradients{length(weights)-layer}(1,j) = weightGradients{length(weights)-layer}(1,j) +  deltas{length(weights)-layer}(j);
            for i=2:size(points,2)+1
                weightGradients{length(weights)}(i,j) = weightGradients{length(weights)}(i,j) + deltas{length(weights)}(j)*points(p,i-1);
            end
        end
    end
end
end



