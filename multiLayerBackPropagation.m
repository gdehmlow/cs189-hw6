function [ weightGradients ] = multiLayerBackPropagation(points,labels,weights,...
    ONLFunc,ONLDerivative,HNLFunc,HNLDerivative,lossDerivative)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

weightGradients = cell(length(weights),1);
deltas = cell(length(weights),1);
%toplayer computation then propagation
for i=1:length(weights)
    weightGradients{i} = zeros(size(weights{i}));
    deltas{i} = zeros(size(weights{i},2),1);
end
for p=1:length(labels)
    labelVector = zeros(size(weights{length(weights)},2),1);
    labelVector(labels(p)+1) = 1;
    [outputs, weightedSums] = multiLayerPredict(points(p,:),weights,ONLFunc,HNLFunc);
    layerDeltas =  zeros(size(weights{length(weights)},2),1);
    layerWeightGradients = weightGradients{length(weightGradients)};
    layerInputs = [1;outputs{length(outputs)-1}];
    layerOutputs = outputs{length(outputs)};
    layerWeightedSums = weightedSums{length(weightedSums)};
    for j=1:length(layerOutputs)
        lossD = lossDerivative(layerOutputs(j),labelVector(j));
        ONLD = ONLDerivative(layerWeightedSums(j));
        layerDeltas(j) = lossD*ONLD;
        layerWeightGradients(:,j) = layerDeltas(j).*layerInputs;
        %layerWeightGradients(1,j) = layerWeightGradients(1,j) +...
         %   layerDeltas(j);
        %for i=2:length(weightedSums{length(weightedSums)-1})+1
         %   layerWeightGradients(i,j) = layerWeightGradients(i,j) +...
          %      layerDeltas(j)*layerInputs(i-1);
        %end
    end
    weightGradients{length(weights)} = layerWeightGradients;
    prevLayerDeltas = layerDeltas;
    for layerDepth=1:length(weights)-1
        layerWeightGradients = weightGradients{length(weightGradients)-layerDepth};
        prevLayerWeights = weights{length(weights)-layerDepth+1};
        layerWeightedSums = weightedSums{length(weightedSums)-layerDepth};
        if layerDepth == length(weights)-1
            layerInputs = [1;points(p,:)'];
        else
            layerInputs = [1;outputs{length(outputs)-layerDepth-1}];
        end
        layerDeltas = (prevLayerWeights(2:length(prevLayerWeights),:)*prevLayerDeltas);
        layerDeltas = layerDeltas.*HNLDerivative(layerWeightedSums);
        %for i=1:length(layerDeltas)
         %   for j=1:size(prevLayerWeights,2)
          %      layerDeltas(i) = layerDeltas(i) +  prevLayerWeights(i,j)*prevLayerDeltas(j);
           % end
            %HNLD = HNLDerivative(layerWeightedSums(j));
       %     if isnan(HNLD)
        %        x=0;
         %   end
          %  layerDeltas(j) = layerDeltas(j)*HNLD;
        %end
        for j=1:length(layerDeltas)
            layerWeightGradients(:,j) = layerWeightGradients(:,j)+layerDeltas(j).*layerInputs;
        end
        %for j=1:size(layerWeightGradients,2)
         %   layerWeightGradients(1,j) = layerWeightGradients(1,j) + layerDeltas(j);
          %  for i=2:length(layerInputs)+1
           %     layerWeightGradients(i,j) = layerWeightGradients(i,j) + layerDeltas(j)*layerInputs(i-1);
           % end
        %end
        prevLayerDeltas = layerDeltas;
        weightGradients{length(weightGradients)-layerDepth} = layerWeightGradients;
    end
end
end



