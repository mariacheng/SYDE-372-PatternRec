% This contains the script for questions 3 and 4 in Lab 3
% Question 3 builds an MICD classifier 
    % This classifier is used on the provided test dataset
    % We then generate the confusion matrix and misclassification rate
% Question 4 uses the same classifier on a second image where each pixels
% corresponding feature values are provided

% Clear the Workspace and Command window - don't delete the provided
% variables if they already exist, if not, load the feat.mat file
clc; close all; clearvars -except f2 f2t f32 f32t f8 f8t multf8 multim
if ~exist('f2', 'var')
    load feat.mat
end

% "Switches" to control plotting the images
% 1: Plot
% 0: Don't Plot
PLOT3 = 0;
PLOT4 = 0;

%% Question 3
% Define a grid where each point is assigned to a class by following the
% MICD decision rule - minimizing the Eucledian distance to the sample mean
% in the transformed feature space

% Grid spacing: smaller = higher resolution
spacing = 0.0005; 
[X, Y] = meshgrid(min(f2(1, :)):spacing:max(f2(1, :)), max(f2(2, :)):-spacing:min(f2(2, :)));

% Initializing the matrices calculating the minimum distance
MICDf2 = ones(numel(X), 1); MICDf8 = MICDf2; MICDf32 = MICDf2;
for i = 1:10
    % Sample means for the 3 cases; n = [2, 8, 32]
    meanf2 = mean([f2(1, (i-1)*16+1 : i*16 ); f2(2, (i-1)*16+1 : i*16)], 2);
    meanf8 = mean([f8(1, (i-1)*16+1 : i*16 ); f8(2, (i-1)*16+1 : i*16)], 2);
    meanf32 = mean([f32(1, (i-1)*16+1 : i*16 ); f32(2, (i-1)*16+1 : i*16)], 2);
    
    % Sample covariances for the 3 cases; n = [2, 8, 32]
    covf2 = cov(f2(1, (i-1)*16+1 : i*16 ), f2(2, (i-1)*16+1 : i*16));
    covf8 = cov(f8(1, (i-1)*16+1 : i*16 ), f8(2, (i-1)*16+1 : i*16));
    covf32 = cov(f32(1, (i-1)*16+1 : i*16 ), f32(2, (i-1)*16+1 : i*16));
    
    % Distance to mean in the transformed feature space = (x-m)'S^-1(x-m) 
    % Each layer (i) of this 3-dimensional matrix has the distance of each 
    % point to the mean of that class (i)
    MICDf2(:, :, i) = sum(bsxfun(@minus, [X(:), Y(:)], meanf2') * ...
                        inv(covf2) .* bsxfun(@minus, [X(:), Y(:)], meanf2'), 2); 
    MICDf8(:, :, i) = sum(bsxfun(@minus, [X(:), Y(:)], meanf8') * ...
                        inv(covf8) .* bsxfun(@minus, [X(:), Y(:)], meanf8'), 2);
    MICDf32(:, :, i) = sum(bsxfun(@minus, [X(:), Y(:)], meanf32') * ...
                        inv(covf32) .* bsxfun(@minus, [X(:), Y(:)], meanf32'), 2);
end
% Each point is then classified to the class where the distance to mean in 
% the transformed feature space is minimized
    % [~, index] = min(MICD, along the 3rd dimension), where index returns  
    % a matrix with the classes each grid point is closest to
% The edge function is used to identify the decision boundaries
[~, classifiedf2] = min(MICDf2, [], 3); clearvars MICDf2; 
classifiedf2 = reshape(classifiedf2, size(X));
[Yf2, Xf2] = find(edge(classifiedf2, 'Canny') == 1);

[~, classifiedf8] = min(MICDf8, [], 3); clearvars MICDf8; 
classifiedf8 = reshape(classifiedf8, size(X));
[Yf8, Xf8] = find(edge(classifiedf8, 'Canny') == 1);

[~, classifiedf32] = min(MICDf32, [], 3); clearvars MICDf32; 
classifiedf32 = reshape(classifiedf32, size(X));
[Yf32, Xf32] = find(edge(classifiedf32, 'Canny') == 1);

if PLOT3 % Plotting decision boundaries
figure
aplot(f2); hold on; plot(X(1,Xf2), Y(Yf2,1), 'k.');hold off; grid on; grid minor
title('f2'); xlabel('x_1'); ylabel('x_2')

figure
aplot(f8); hold on; plot(X(1,Xf8), Y(Yf8,1), 'k.');hold off; grid on; grid minor
title('f8'); xlabel('x_1'); ylabel('x_2')

figure
aplot(f32); hold on; plot(X(1,Xf32), Y(Yf32,1), 'k.');hold off; grid on; grid minor
title('f32'); xlabel('x_1'); ylabel('x_2')
end

% Confusion Matrix
% Initializing the confusion matrices
confMatrixf2 = zeros(10); confMatrixf8 = zeros(10); confMatrixf32 = zeros(10);
for i = 1:160
    
    % Apply the classifier to the test data
    % Identify the grid point closest to the test data and assign the test
    % data to the class of the grid point
    % Update the confusion matrix by adding 1 to the appropriate element
    [~, yy] = min(abs(f2t(1, i) - X(1, :))); [~, xx] = min(abs(f2t(2, i)-Y(:, 1)));
    confMatrixf2(f2t(3, i), classifiedf2(xx, yy)) = confMatrixf2(f2t(3, i), classifiedf2(xx, yy)) + 1;
    
    [~, yy] = min(abs(f8t(1, i) - X(1, :))); [~, xx] = min(abs(f8t(2, i)-Y(:, 1)));
    confMatrixf8(f8t(3, i), classifiedf8(xx, yy)) = confMatrixf8(f8t(3, i), classifiedf8(xx, yy)) + 1;
    
    [~, yy] = min(abs(f32t(1, i) - X(1, :))); [~, xx] = min(abs(f32t(2, i)-Y(:, 1)));
    confMatrixf32(f32t(3, i), classifiedf32(xx, yy)) = confMatrixf32(f32t(3, i), classifiedf32(xx, yy)) + 1;
end
% Misclassification Rate (%)
% The diagonal of the confusion matrix contains the correctly classified
% values; the sum of the off-diagonal terms, along the row, is the total 
% misclassified points for that particular class
MCRf2 = sum(confMatrixf2 - diag(diag(confMatrixf2)), 2)' * 100 / 16;
MCRf8 = sum(confMatrixf8 - diag(diag(confMatrixf8)), 2)' * 100 / 16;
MCRf32 = sum(confMatrixf32 - diag(diag(confMatrixf32)), 2)' * 100 / 16;

%% Question 4
% Similar to before, match the feature values to the closest grid point in
% the classifier space, similar to how it was done in question 3, to 
% classify each pixel and assign that to cimage
[~, yy] = min(abs(bsxfun(@minus, multf8(:, :, 1), reshape(X(1, :), [1, 1, length(X(1, :))]))), [], 3);
[~, xx] = min(abs(bsxfun(@minus, multf8(:, :, 2), reshape(Y(:, 1), [1, 1, length(Y(:, 1))]))), [], 3);
cimage = reshape(classifiedf8(sub2ind(size(classifiedf8), xx(:), yy(:))), [256, 256]);

if PLOT4
figure
imagesc(multim); colorbar; xlabel('x_1'); ylabel('x_2')

figure
imagesc(cimage); colorbar; xlabel('x_1'); ylabel('x_2')
end
