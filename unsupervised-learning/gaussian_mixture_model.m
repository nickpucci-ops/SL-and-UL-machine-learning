% gaussian_mixture_model.m
clear; clc;

dataA = readmatrix('data/DatasetA.csv');
dataB = readmatrix('data/DatasetB.csv');

dataset_choice = input('Enter dataset to analyze (A or B): ', 's');
switch upper(dataset_choice)
    case 'A'
        X = dataA;
        dataset_name = 'DatasetA';
    case 'B'
        X = dataB;
        dataset_name = 'DatasetB';
    otherwise
        error('Invalid choice. Please enter A or B.');
end

choice = input('Specify k manually (M) or let GMM predict (P)? [M/P]: ', 's');
if upper(choice) == 'M'
    k = input('Enter number of clusters (k): ');
    if ~isscalar(k) || k < 1 || floor(k) ~= k
        error('k must be a positive integer.');
    end
else
    % Predict optimal k using Bayesian Information Criterion scoring
    max_k = 10;  % Max clusters to test (adjust as needed)
    bic = zeros(max_k, 1); %array to store BIC values for each k
    gm_models = cell(max_k, 1); %Create a cell array to store GMM models for each k
    
    fprintf('Testing k from 1 to %d...\n', max_k);
    for k = 1:max_k
        % Fit a GMM with k clusters to the data X
        % 'RegularizationValue' adds a small value to the covariance to prevent errors
        % 'MaxIter' sets the maximum number of iterations for the algorithm to converge
        gm = fitgmdist(X, k, 'RegularizationValue', 0.01, 'Options', statset('MaxIter', 100));
        bic(k) = gm.BIC;  % Store the BIC score for this k (lower BIC means a better fit)
        gm_models{k} = gm; % Save the GMM model for this k in the cell array
    end
    
    % Find k with minimum BIC
    [~, k_opt] = min(bic);% Get the index of the minimum BIC (k_opt is the optimal k)
    fprintf('Predicted optimal k for %s: %d (based on BIC)\n', dataset_name, k_opt);
    
    % Plot BIC to visualize
    figure;
    plot(1:max_k, bic, '-o');% Plot BIC scores for each k with a line and dots
    hold on;% Keep the plot active to add more elements
    plot(k_opt, bic(k_opt), 'r*', 'MarkerSize', 10); % Mark the optimal k with a red star
    title(sprintf('BIC vs. k for %s', dataset_name));
    xlabel('Number of Clusters (k)');
    ylabel('BIC');
    hold off;
    
    k = k_opt;  % Set k to the optimal value found by BIC (the predicted k)
end

% Fit a final GMM model using the selected k, with the same regularization and iteration settings
gm = fitgmdist(X, k, 'RegularizationValue', 0.01, 'Options', statset('MaxIter', 100));

% Predict cluster assignments
% Assign each data point to a cluster based on the highest probability from the GMM
cluster_idx = cluster(gm, X);

fprintf('\nResults for %s with k = %d:\n', dataset_name, k);
fprintf('Number of Clusters: %d\n', k);
fprintf('Cluster Means:\n');
disp(gm.mu); % Display the means (centroids) of each cluster
fprintf('Cluster Covariances:\n');
for i = 1:k
    fprintf('Cluster %d:\n', i);
    disp(gm.Sigma(:,:,i)); % the covariance matrix for this cluster
end
fprintf('Mixing Proportions:\n');
disp(gm.ComponentProportion); %the mixing proportions (how much each cluster contributes)

figure;
scatter(X(:,1), X(:,2), 10, cluster_idx, 'filled');
hold on;
for i = 1:k
    mu = gm.mu(i,:); % Get the mean (center) of cluster i (a 1x2 vector: [x, y])
    Sigma = gm.Sigma(:,:,i);% Get the covariance matrix of cluster i (a 2x2 matrix)
    % Compute the eigenvectors (V) and eigenvalues (D) of the covariance matrix
    % V gives the directions of the ellipse axes, D gives the lengths (scaled by eigenvalues)
    [V, D] = eig(Sigma);
    % Create points for a unit circle (to transform into an ellipse)
    % linspace(0, 2*pi, 100) makes 100 points evenly spaced around a circle
    t = linspace(0, 2*pi, 100);
    % Transform the unit circle into an ellipse:
    % [cos(t); sin(t)] are points on a unit circle
    % sqrt(D) scales the circle by the square root of the eigenvalues (ellipse axes lengths)
    % V rotates the ellipse to align with the covariance directions
    % * 1.96 scales the ellipse to cover 95% of the data (for a Gaussian, 1.96 std devs)
    ellipse = (V * sqrt(D) * [cos(t); sin(t)])' * 1.96;  % 95% CI
    % Plot the ellipse:
    % Add the ellipse points to the cluster mean (mu(1), mu(2)) to center it
    % 'k-' means a black line, 'LineWidth', 1.5 makes it thicker for visibility
    plot(mu(1) + ellipse(:,1), mu(2) + ellipse(:,2), 'k-', 'LineWidth', 1.5);
end
hold off;
title(sprintf('GMM Clustering for %s (k = %d)', dataset_name, k));
xlabel('Feature 1');
ylabel('Feature 2');
colormap('jet');
colorbar;