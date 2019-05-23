function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

%Utilize the O(n^3) normal eqn to avoid gradient descent. Prefer to use
%this until around N = 10,000 samples

%pinv is psuedo inverse, in case matrix doesnt have inverse
theta = pinv(X'*X) * X' * y;

% -------------------------------------------------------------


% ============================================================

end
