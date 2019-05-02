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