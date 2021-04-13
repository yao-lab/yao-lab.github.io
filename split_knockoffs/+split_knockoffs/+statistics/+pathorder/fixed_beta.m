function beta = fixed_beta(X, y, D, option)
% split_knockoffs.statistics.pathorder.fixed_beta calculate the fixed beta
% for fixed beta split Knockoffs. 

% input argument
% X : the design matrix
% y : the response vector
% D : the linear transform
% option: options for creating the Knockoff statistics
% % option.beta : the choice of fixed beta for step 0: 
% % % % "mle" maximum likelihood estimator; 
% % % % "cv_split" cross validation choice of split LASSO over nu and lambda
% % % % "cv_ridge" cross validation choice of ridge regression over lambda

% output argument
% beta: the choice of beta based on the option.

beta_choice = option.beta;

switch beta_choice
    case "mle"
        beta = (X' * X)^(-1) * X' * y;
    case "cv_split"
        beta = split_knockoffs.statistics.pathorder.cv_split(X, y, D);
    case "cv_ridge"
        beta = split_knockoffs.statistics.pathorder.cv_ridge(X, y, D);
end
end