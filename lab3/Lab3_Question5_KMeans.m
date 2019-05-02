clear all
close all
load feat.mat

% Initialization
K = 10;
data = f32(1:2,:);
prototypes = datasample(data, K, 2);
prototypes = prototypes(1:2,:)';
data = data';
old_prototypes = zeros(K,1);

% Keep iterating until prototypes don't change
while true
    old_prototypes = prototypes;
    
    % Determine min distance, reassign clusters
    [~, idx] = pdist2(prototypes,data,'euclidean','Smallest',1);
    for i = 1:K
        id = idx == i;
        prototypes(i,:) = nanmean(data(id,:)); % Recalculate mean
    end
    
    if old_prototypes == prototypes; break; end

end

% Plot
gscatter(data(:,1),data(:,2),idx);
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8', 'Cluster 9', 'Cluster 10');
title('K means Algorithm converged with K = 10');
ylabel('x_2');
xlabel('x_1');