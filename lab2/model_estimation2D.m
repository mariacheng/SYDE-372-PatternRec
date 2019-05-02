load('lab2_2.mat')

%Gaussian
a_est = gaussian_estimate2D(al);
b_est = gaussian_estimate2D(bl);
c_est = gaussian_estimate2D(cl);
figure;

plot(al(:,1),al(:,2),'.');
hold on
plot(bl(:,1),bl(:,2),'.');
hold on
plot(cl(:,1),cl(:,2),'.');
hold on

plotMAP({a_est{1}, b_est{1}, c_est{1}}, [al; bl; cl], 300 ,{a_est{2},b_est{2},c_est{2}}, [100, 100, 100],'MAP');
title('Parametric Estimation 2D - Gaussian');
legend('a','b','c','boundary');
ylabel('x_2')
xlabel('x_1')
%Parzen
win = fspecial('gaussian',[10 10], 20);
% x = -10:0.1:10;
% [X1, X2] = meshgrid(x,x);
% X = [X1(:) X2(:)];
% y = mvnpdf(X, 0, 20);

[p_a,x,y] = parzen(al, [5 0 0 450 450], win);
[p_b,x,y] = parzen(bl, [5 0 0 450 450], win);
[p_c,x,y] = parzen(cl, [5 0 0 450 450], win);
[X,Y] = meshgrid(x,y);
BW_a = edge(p_a > 0);
BW_b = edge(p_b > 0);
BW_c = edge(p_c > 0);
plot(X(BW_a),Y(BW_a),'.');
hold on
plot(X(BW_b),Y(BW_b),'.');
hold on
plot(X(BW_c),Y(BW_c),'.');
hold on
title('Non-Parametric Estimation 2D - Parzen');
legend('a bound','b bound','c bound')
ylabel('x_2')
xlabel('x_1')
function params = gaussian_estimate2D(x)
    n = length(x);
    params{1} = mean(x);
    params{2} = ((n-1)/n)*cov(x, 1);
end