function[result, Z] = convert_knockoff(X, D, y, option)
% split_knockoffs.private.convert_knockoff convert structural problem to
% traditional sparse support set recover problem, using the orthogonal
% complement. This function uses equal design for knockoff.

% input arguments
% X : the design matrix
% y : the response vector
% D : the linear transform
% option.q : the desired FDR control bound
% option.method: "Knockoff" or "Knockoff+"

% output arguments
% result: the estimated support set
% Z: [Z, tilde_Z] used for knockoffs

[m, p] = size(D);

% calculate X_new
X_new = X * pinv(D);
y_new = y;

% multiply the  orthogonal complement
if m < p
    X_0 = X * null(D);
    Ortho = null(X_0');
    X_new = Ortho' * X_new;
    y_new = Ortho' * y_new;
end

[result, ~, Z] = knockoffs.filter(X_new, y_new, option.q, {'fixed'}, 'Method', 'equi', 'Threshold', option.method);
end