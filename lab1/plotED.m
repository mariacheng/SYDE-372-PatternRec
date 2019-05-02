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