 % SYDE Lab 1 - Clusters and Classification Boundaries
% Name: Maria Cheng Date: Jan 26, 2019
clear all
close all

n = 250;% num descritization
k  = 5; % k points for NN

% define mean and variance of pdf
mu_A = [5 10];
sigma_A = [8 0; 0 4];
n_A = 200;
x_A = repmat(mu_A,n_A,1) + randn(n_A,2)*chol(sigma_A);

p=plot(x_A(:,1),x_A(:,2),'b.');
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
plotContour(mu_A,sigma_A);

mu_B = [10 15];
x_B = repmat(mu_B,n_A,1) + randn(n_A,2)*chol(sigma_A); % range of the random variable x2

hold on
p=plot(x_B(:,1),x_B(:,2),'r.');
plotContour(mu_B,sigma_A);
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');


% NN and kNN Test points
test_A = repmat(mu_A,n_A,1) + randn(n_A,2)*chol(sigma_A);
test_B = repmat(mu_B,n_A,1) + randn(n_A,2)*chol(sigma_A);

%%% Draw plots
% TODO: Figure out how to put it in the legend
% ED_labels = plotED({mu_A, mu_B}, [x_A; x_B], n,'','MED');
% ED_confmat = calcConfMat(ED_labels, {x_A, x_B});
% GED_labels = plotED({mu_A, mu_B}, [x_A; x_B], n, {sigma_A, sigma_A},'MICD');
% GED_confmat = calcConfMat(GED_labels, {x_A, x_B});
% MAP_labels = plotMAP({mu_A, mu_B}, [x_A; x_B], n, {sigma_A, sigma_A}, [n_A, n_A],'MAP');
% MAP_confmat = calcConfMat(MAP_labels, {x_A, x_B});
% axis equal
% legend
% title('Case 1 - MED, MICD, MAP Classifiers');

% NN
% NN_labels = plotkNN({x_A x_B}, n, 1,'NN');
% NN_confmat = calcConfMat(NN_labels, {test_A, test_B});
% kNN_labels = plotkNN({x_A x_B}, n, k,'kNN, k=5');
% kNN_confmat = calcConfMat(kNN_labels, {test_A, test_B});
%  
% axis equal
% legend
% title('Case 1 - NN, kNN Classifiers');


% Case 2
n_C = 100;
mu_C = [5 10];
sigma_C = [8 4; 4 40];
x_C = repmat(mu_C,100,1) + randn(100,2)*chol(sigma_C);

figure;
p=plot(x_C(:,1),x_C(:,2),'b.');
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
plotContour(mu_C,sigma_C);

n_D = 200;
mu_D = [15 10];
sigma_D = [8 0; 0 8];
x_D = repmat(mu_D,200,1) + randn(200,2)*chol(sigma_D);

hold on
p=plot(x_D(:,1),x_D(:,2),'r.');
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
plotContour(mu_D,sigma_D);

n_E = 150;
mu_E = [10 5];
sigma_E = [10 -5; -5 20];
x_E = repmat(mu_E,150,1) + randn(150,2)*chol(sigma_E);

hold on
p=plot(x_E(:,1),x_E(:,2),'.');
set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
plotContour(mu_E,sigma_E);

%%%% DRaw plots
ED_3labels = plotED({mu_C, mu_D, mu_E}, [x_C; x_D; x_E], n,'','MED');
% ED_3confmat = calcConfMat(ED_3labels, {x_C, x_D, x_E});
GED_labels = plotED({mu_C, mu_D, mu_E}, [x_C; x_D; x_E], n,{sigma_C,sigma_D,sigma_E},'MICD'); %GED
plotMAP({mu_C, mu_D, mu_E}, [x_C; x_D; x_E], n,{sigma_C,sigma_D,sigma_E}, [n_C, n_D, n_E],'MAP');
title('Case 2 - MED, MICD, MAP Classifiers');

plotkNN({x_C x_D x_E}, n, 1,'NN');
plotkNN({x_C x_D x_E}, n, k,'kNN, k=5');
title('Case 2 - NN, kNN Classifiers');

axis equal
legend
function plotContour(Mu,Sigma)
    [V,D] = eig(Sigma);
    t=-pi:0.01:pi;
    x = sqrt(D(1,1))*cos(t);
    y = sqrt(D(2,2))*sin(t);
    % Rotate ellipse
    rotated = V*[x;y];
    hold on
    x = Mu(1) + rotated(1,:);
    y = Mu(2) + rotated(2,:);
    p=plot(x,y);
    set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end

function label_region = plotED(Mu, X, n, Sigma,plotName)
    for a = 1:length(Mu)
        if isempty(Sigma) % for MED
            W{a} = 1;
        else
            W{a} = inv(Sigma{a});
        end
    end

    xmin = floor(min(X(:,1)));
    xmax = ceil(max(X(:,1)));
    ymin = floor(min(X(:,2)));
    ymax = ceil(max(X(:,2)));
    
    x = linspace(xmin, xmax, n);
    y = linspace(ymin, ymax, n);
    [X,Y] = meshgrid(x,y);
    
    for a = 1:length(Mu)
        diff_mu = bsxfun(@minus,[X(:) Y(:)], Mu{a});
        for i = 1:size(diff_mu,1)
            mu_dist{a}(i) = sqrt(diff_mu(i,:)*W{a}*diff_mu(i,:)');
        end
    end
    
    % Determine mask and boundary
    [M, I] = min(cell2mat(mu_dist'),[],1);
    mask = reshape(I,size(X));
    BW = edge(mask);
    plot(X(BW),Y(BW),'.','DisplayName',plotName);
    hold on

    for a = 1:length(Mu)
        label_region{a} = [X(mask==a), Y(mask==a)];
    end
end

function label_region = plotkNN(X, n, k,plotName)
    for a = 1:length(X)
        xp{a} = X{a};
    end
    X = cell2mat(X');
    xmin = floor(min(X(:,1)));
    xmax = ceil(max(X(:,1)));
    ymin = floor(min(X(:,2)));
    ymax = ceil(max(X(:,2)));
    
    x = linspace(xmin, xmax, n);
    y = linspace(ymin, ymax, n);
    [X,Y] = meshgrid(x,y);
    
    for a = 1:length(xp)
        % calc pairwise distance across all class points
        % find the min k points and calculate mean across all rows
        class{a} = mean(mink(pdist2([X(:),Y(:)],xp{a}),k,2),2); % default Euclidean distance
    end
    % minimum between the classes
    [M, I] = min(cell2mat(class),[],2);
    mask = reshape(I,size(X));
    BW = edge(mask);
    hold on
    plot(X(BW),Y(BW),'.','DisplayName',plotName);
    
    for a = 1:length(xp)
        label_region{a} = [X(mask==a), Y(mask==a)];
    end
end

function label_region = plotMAP(Mu, X, n, Sigma, N_points,plotName)
    xmin = floor(min(X(:,1)));
    xmax = ceil(max(X(:,1)));
    ymin = floor(min(X(:,2)));
    ymax = ceil(max(X(:,2)));
    
    x = linspace(xmin, xmax, n);
    y = linspace(ymin, ymax, n);
    [X,Y] = meshgrid(x,y);

    for a = 1:length(Mu)
        invSig = inv(Sigma{a});
        diff_mu = bsxfun(@minus,[X(:) Y(:)], Mu{a});
        prior = N_points(a)./sum(N_points);
        for i = 1:size(diff_mu,1)
            dist(i) = diff_mu(i,:)*invSig*diff_mu(i,:)';
        end
        t = log(prior) - log(2*pi) - 1/2*log(det(Sigma{a}));
        class{a} = -1/2*dist + repmat(t, size(dist));
    end
    [M, I] = max(cell2mat(class'),[],1);
    mask = reshape(I,size(X));
    BW = edge(mask);
    hold on
    plot(X(BW),Y(BW),'.','DisplayName',plotName);
    
    for a = 1:length(Mu)
        label_region{a} = [X(mask==a), Y(mask==a)];
    end
end

function cmat = calcConfMat(labels, datapoints)
    actual_label = [];
    pred_label = [];

    % Multi-class
    for a = 1:length(datapoints)
        min_dist = [];
        for b = 1:length(labels) % Multi-class calculation
            min_dist = [min_dist; min(pdist2(labels{b},datapoints{a}))]; % determine min distances between data points and 'class region points'
        end
%         min_dist = [min(pdist2(labels{1},datapoints{a})); min(pdist2(labels{2},datapoints{a}))];
        [m,label] = min(min_dist); % which region does the point reside
        pred_label = [pred_label, label];
        actual_label = [actual_label, ones(1,length(datapoints{a}))*a];
    end
    % calculate the confusion matrix with known labels and predicted labels
    cmat = confusionmat(actual_label, pred_label);
end
