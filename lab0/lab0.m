% SYDE Lab 0 - MATLAB Introduction
% Name: Maria Cheng Date: Jan 26, 2019

clear all
close all

% define mean and variance of pdf
mu = [0 0]';
sigma = [1 0; 0 1];

dx = 0.1; % step-size
x1 = [-3:dx:3]; % range of the random variable x1
x2 = [-3:dx:3]; % range of the random variable x2

% call function
y = Gauss2d(x1,x2,mu,sigma);

% Show 3-D plot of pdf
figure;
subplot(2,1,1);
surf(x1,x2,y);
xlabel('x_1'); ylabel('x_2');

% Show contours of pdf
subplot(2,1,2);
contour(x1,x2,y);
xlabel('x_{1}'); ylabel('x_{2}');
axis equal

% Show colour map of pdf
figure;
imagesc(x1,x2,y);
xlabel('x_{1}'); ylabel('x_{2}');

z = (y > 0.1);
figure; imagesc(x1,x2,z);
hold on
plot(mu(1,1),mu(2,1),'gx');
xlabel('x_{1}'); ylabel('x_{2}');
