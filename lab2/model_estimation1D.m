load('lab2_1.mat')
a_gauss = gaussian_estimate(a);
b_gauss = gaussian_estimate(b);
figure;
plot_PDF('Gaussian', {5, 1})
hold on
plot_PDF('Gaussian', a_gauss)
hold on
plot_PDF('Exponential', 1)
hold on
plot_PDF('Gaussian', b_gauss)
hold on
title('Gaussian parametric estimation');
legend('true p(x) for a', 'p(x) estimate for a', 'true p(x) for b', 'p(x) estimate for b');
ylabel('p(x)')
xlabel('x')

a_exp = exp_estimate(a);
b_exp = exp_estimate(b);
figure;
plot_PDF('Gaussian', {5, 1});
hold on
plot_PDF('Exponential', a_exp)
hold on
plot_PDF('Exponential', 1)
hold on
plot_PDF('Exponential', b_exp)
hold on
title('Exponential parametric estimation');
legend('true p(x) for a', 'p(x) estimate for a', 'true p(x) for b', 'p(x) estimate for b');
ylabel('p(x)')
xlabel('x')

a_uniform = uniform_estimate(a);
b_uniform = uniform_estimate(b);
figure;
plot_PDF('Gaussian', {5, 1})
hold on
plot_PDF('Uniform', a_uniform);
hold on
plot_PDF('Exponential', 1)
hold on
plot_PDF('Uniform', b_uniform);
hold on
title('Uniform parametric estimation');
legend('true p(x) for a','uniform p(x) estimate for a', 'true p(x) for b','uniform p(x) estimate for b');
ylabel('p(x)')
xlabel('x')

figure;
nonparametric_estimate(a, 0.1);
hold on
plot_PDF('Gaussian', {5, 1});
hold on
nonparametric_estimate(b, 0.1);
hold on
plot_PDF('Exponential', 1);
title('Parzen Window Estimation - std 0.1')
legend('true p(x) for a', 'nonparametric estimate for a', 'true p(x) for b', 'nonparametric estimate for b');
ylabel('p(x)')
xlabel('x')

figure;
nonparametric_estimate(a, 0.4);
hold on
plot_PDF('Gaussian', {5, 1});
hold on
plot_PDF('Exponential', 1);
hold on
nonparametric_estimate(b, 0.4);
hold on
title('Parzen Window Estimation - std 0.4')
legend('true p(x) for a', 'nonparametric estimate for a', 'true p(x) for b', 'nonparametric estimate for b');
ylabel('p(x)')
xlabel('x')

% Parametric Estimation with MLE
function params = gaussian_estimate(x)
    % x is the sample inputs
    % params is the output
    
    N = length(x);
    params{1} = sum(x)/N;
    params{2} = sum((x-params{1}).^2)/N;

end

function params = exp_estimate(x)
    % x sample inputs
    % params estimated output
    
    N = length(x);
    params = N/sum(x);

end

function params = uniform_estimate(x)
    % x sample inputs
    % params estimated params a and b
    
    params{1} = min(x);
    params{2} = max(x);
end

function nonparametric_estimate(x, h)
    % Parzen window estimation
    y = linspace(-5, 10, 300);
    for j = 1:length(y)
        N = length(x);
        p(j) = (1/N)*sum(exp(-1/2*((y(j)-x)/h).^2)/(sqrt(2*pi)*h));
    end
    plot(y, p)
end

function plot_PDF(pdf, params)
    x = linspace(-5, 10, 300);
    if strcmp(pdf, 'Gaussian')
        m = params{1};
        s = params{2};
        
        y = (1/sqrt(2*pi)*s)*exp(-1/2*((x-m)/s).^2);
    elseif strcmp(pdf, 'Exponential')
        x = linspace(0, 10, 150);
        lambda = params;
        y = zeros(length(x));
        for i = 1:length(x)
            if x(i) > 0
                y = exppdf(x, 1/lambda);
            end
        end
    elseif strcmp(pdf, 'Uniform')
        a = params{1};
        b = params{2};
        y = zeros(length(x));
        for i = 1:length(x)
            if x(i) >= a && x(i) <= b
                y(i) = 1/(b-a);
            end
        end
    end
    
    plot(x, y)
    
end