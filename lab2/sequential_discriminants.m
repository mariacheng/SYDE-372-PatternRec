clear all
close all
load('lab2_3.mat')

figure;
plot(a(:,1),a(:,2),'.');
hold on
plot(b(:,1),b(:,2),'.');
hold on
for i = 1:20
    load('lab2_3.mat')
    a_orig = a;
    b_orig = b;
    j = 1;
    na_B(j) = 1;
    nb_A(j) = 1;
    ED_1 = cell(1,6);
    while na_B(j) ~= 0 && nb_A(j) ~= 0
        a_rand = a(randi(size(a,1)),:);
        b_rand = b(randi(size(b,1)),:);

        ED_1{j} = plotED({a_rand, b_rand}, [a; b], 200,'', 'MED');
        [ED_confmat, actual_label, pred_label] = calcConfMat(ED_1{j}, {a, b});
        nb_A(j) = ED_confmat(2,1);
        na_B(j) = ED_confmat(1,2);
        if na_B(j) == 0 || nb_A(j) == 0
            if na_B(j) == 0
                b = b((actual_label{2} - pred_label{2})~=0,:);
                misclass = b_orig;
            end
            if nb_A(j) == 0
                a = a((actual_label{1} - pred_label{1})~=0,:);
                misclass = a_orig;
            end
            %err(j, i) = (nb_A(j) + na_B(j))/(size(a,1)+size(b,1));
            err(j,i) = classify_points(ED_1, [a_orig;b_orig], na_B, nb_A);
            j = j + 1;
            
        end
        if isempty(a) && isempty(b) || j == 6
            na_B(j) = 0;
            nb_A(j) = 0;
        else
            na_B(j) = 1;
            nb_A(j) = 1;
        end
    end
end
disp('hi')
err_rate{1} = mean(err,2);
err_rate{2} = min(err,[],2);
err_rate{3} = max(err,[],2);
err_rate{4} = std(err,1,2);
figure;
plot(1:length(err_rate{1}), err_rate{1})
hold on
plot(1:length(err_rate{1}), err_rate{2})
hold on
plot(1:length(err_rate{1}), err_rate{3})
hold on
plot(1:length(err_rate{1}), err_rate{4})
xlabel('J')
ylabel('Error rate')
legend('mean','min','max','std')
title('Error rate of sequential classifier from J=1:5')

run_seq_discriminators(ED_1, na_B, nb_A);
title('Sequential Classifier 3')
legend('a','b','boundary')
ylabel('x_2')
xlabel('x_1')

function run_seq_discriminators(discriminators,na_B, nb_A)
    n = 100;
    x = linspace(50, 600, n);
    y = linspace(0, 450, n);    
    label = zeros(n);
    for i = 1:n
        for k = 1:n
            j = 1;
            while label(i,k) == 0
                G_j = discriminators{j};
                min_dist = [];
                for b = 1:length(G_j) 
                    min_dist = [min_dist; min(pdist2(G_j{b},[x(k) y(i)]))]; % determine min distances between data points and 'class region points'
                end
                [~, reg] = min(min_dist); % region the point resides in
                if reg == 2 && na_B(j) == 0
                    label(i,k) = 2;
                elseif reg == 1 && nb_A(j) == 0
                    label(i,k) = 1;
                else
                    j = j+ 1;
                end
            end
        end
    end
    [X,Y] = meshgrid(x,y);
    BW = edge(label);
    plot(X(BW),Y(BW),'.');
    hold on
    
end
function err = classify_points(discrim, points, na_B, nb_A)
    lab = zeros(1,length(points));
    for p = 1:length(points)
        j = 1;
        while lab(p) == 0
            G_j = discrim{j};
            min_dist = [];
            for b = 1:length(G_j)
                min_dist = [min_dist; min(pdist2(G_j{b},points(p,:)))]; % determine min distances between data points and 'class region points'
            end
            [~, reg] = min(min_dist);
            if reg == 2 && na_B(j) == 0
                lab(p) = 2;
            elseif reg == 1 && nb_A(j) == 0
                lab(p) = 1;
            elseif ~isempty(discrim{j+1})
                j = j+ 1;
            else
                lab(p) = -1;
            end
        end
    end
    err = sum(lab==-1)/400;
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
%     plot(X(BW),Y(BW),'.','DisplayName',plotName);
%     hold on

    for a = 1:length(Mu)
        label_region{a} = [X(mask==a), Y(mask==a)];
    end
end

function [cmat, actual_label, pred_label] = calcConfMat(labels, datapoints)
    actual_label = {};
    pred_label = {};

    % Multi-class
    for a = 1:length(datapoints)
        min_dist = [];
        for b = 1:length(labels) % Multi-class calculation
            min_dist = [min_dist; min(pdist2(labels{b},datapoints{a}))]; % determine min distances between data points and 'class region points'
        end
%         min_dist = [min(pdist2(labels{1},datapoints{a})); min(pdist2(labels{2},datapoints{a}))];
        [m,label] = min(min_dist); % which region does the point reside
        pred_label{a} = label;
        actual_label{a} = ones(1,size(datapoints{a}, 1))*a;
    end
    % calculate the confusion matrix with known labels and predicted labels
    cmat = confusionmat(cell2mat(actual_label), cell2mat(pred_label));
end