function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

%Create colum-vector solutions to each sample. 9 0's 1 '1'
y_vec = zeros(num_labels, m);
for i=1:m
    y_vec(y(i),i) = 1;
end

% Creates a 25x5000 matrix
hidden_layer = sigmoid(Theta1*[ones(m,1) X]');
% Creates a 10x5000 matrix
output_layer = sigmoid(Theta2*[ones(1,m); hidden_layer]);


J = (1/m) * sum( sum( -y_vec .* log(output_layer) - (1 - y_vec) .* log(1 - output_layer) ));


J = J + lambda/(2*m) .* (sum( sum(Theta1.^2)) + sum( sum(Theta2.^2)));
J = J - lambda/(2*m) .* (sum( sum(Theta1(:,1).^2) + sum( sum(Theta2(:,1).^2))));

% [m, h] = max(output_la.yer, [], 1);

a_1 = 0;
a_2 = 0;
a_3 = 0;
z_2 = 0;
z_3 = 0;
delta_3 = 0;
delta_2 = 0;


DELTA_2 = 0;
DELTA_1 = 0;
in = X';
for t = 1:m
    %Perform forward prop
    a_1 = [1; in(:, t)]; %401x1
    z_2 = Theta1 * a_1; %25x1
    a_2 = [1; sigmoid(z_2)];%26x1
    z_3 = Theta2 * a_2;%10x1
    a_3 = sigmoid(z_3);%10x1
   
    
    %Perform backprop
    delta_3 = a_3 - y_vec(:,t);
    delta_2 = ((Theta2)' * delta_3).* sigmoidGradient([1; z_2]);
    
    %Accumulate the gradient
    DELTA_2 = DELTA_2 + (delta_3) * a_2';
    DELTA_1 = DELTA_1 + (delta_2(2:end)) * a_1';
end

%In order to calculte gradient, must consider how regularization affects. 
%Essentially, we can ignore first columns of Thetas, as they carry the bias
%weights
Theta1_grad = (1/m) * DELTA_1; %25x401
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad = (1/m) * DELTA_2; %10x26
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
