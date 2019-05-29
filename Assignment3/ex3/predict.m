function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Objective: make predictions using your learned neural network. Set p to a 
% vector containing labels between 1 to num_labels.

%Calculate hidden later, adds bias to X
hidden_layer = sigmoid( Theta1 * [ones(m,1) X]' ); 

%Calculates the output later (1-num_labels size), adds bias to hidden
%layer

%Calculates the output layer, adds the bias to the hidden layer
output_layer = sigmoid( Theta2 * [ones(1,m); hidden_layer]);

%Finds max sigmoid value and index per training data
[m, p] = max( output_layer, [], 1);
p = p';


end
