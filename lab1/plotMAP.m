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