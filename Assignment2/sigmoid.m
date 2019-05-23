function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

% Computes the sigmoid value of 'z'. 'z' can be a scalar, vector, or
% matrix. If Z has multiple elements, the sigmoid function is applied to
% each element. 

g = 1 ./(1 + exp(-z));

end
