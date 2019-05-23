function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

%To be returned
J = 0;
grad = zeros(size(theta));


%Compute the cost of each particular theta, set to J. Compute the partial
%derivatives w.r.t. each parameter in theta, set to grad.

%logistic hypothesis
h = sigmoid(X * theta);

J = -1/m * (y' * log(h) + (1-y)' * log(1-h));

%Gradient for fminunc
grad = 1/m * (X' * (h - y));


end
