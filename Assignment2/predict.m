function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

%Variable to be returned
p = zeros(m, 1);


%Making predictions using the learned logistic regression parameters. Turns
%result into column vector of 1's (admitted) and 0's (not admitted)

p = sigmoid(X * theta) >= 0.5;

end
