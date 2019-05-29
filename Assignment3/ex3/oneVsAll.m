function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


% Objective: train num_labels logistic regression classifiers with 
% regularization parameter lambda. 


%Prepare initial theta values for fmincg
initial_thetas = zeros(n + 1, 1);

%set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

%Iterate though all labels, 10 corresponds to 0. Set resulting theta
%reductions to row i in all_theta.
for i=1:num_labels
    all_theta(i,:) = ...
        fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
            initial_thetas, options);
end

end
