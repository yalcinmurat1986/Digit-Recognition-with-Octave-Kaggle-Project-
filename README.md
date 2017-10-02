# Digit-Recognition-with-Octave-Kaggle-Project-


pkg install -forge io  # Required to install csv file
pkg load io             # Required to install csv file
train = csvread('train.csv');  # load files
test=csvread('test.csv');

y=train(:,1);  # take out y
X=train(:,(2:785));   # take out X
m=size(X,1);

%%  X = [(ones(m, 1)) X]; # add 1's to X
alpha = 0.02;  # set alpha
num_iters = 400;  # set number of iterations

#######  Teaching with gradient descent ################################
n=size(X,2);
theta = zeros(n, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
#######  Teaching with gradient descent ################################


%%%%%%%%%% Teaching multi class clasification with Gradient Descent %%%%%%%%%%%% 
input_layer_size  = 784;  % 28x28 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10

%%%%%%%% Display DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);
%%%%%%%% Display DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
fprintf('\nCost: %f\n', J);
fprintf(' %f \n', grad);

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

#### Test Theta with Train Data ##################################
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
#### Test Theta with Train Data ##################################

%%%%%%%%%%%%% test it with Test Data #############################
X=test;
pred = predictOneVsAll(all_theta, X);
csvwrite ("pred2.csv", pred);
%%%%%%%%%%%%% test it with Test Data #############################
