% (a)
X = X';
Y = log(X);
% (b)
for i = 1:452
    for t = 1:1257
        delta_Y(i,t) = Y(i,t+1)-Y(i,t);
    end
end
% (c)
for i = 1:452
    for j = 1:452
        total = 0;
        for tau = 1:1257
            total = total + delta_Y(i,tau)*delta_Y(j,tau);
        end
        Sigma(i,j) = total/1257;
    end
end
% (d)
[V,D] = eig(Sigma);
[D, ind] = sort(diag(D),'descend');
V = V(:, ind);
% (e)
R = 10;
for r = 1:R
    delta_Y_copy = delta_Y;
    for i = 2:452
        rand_seed = randperm(1257);
        delta_Y_copy(i,:) = delta_Y_copy(i,rand_seed);
    end
    Sigma_tilde = delta_Y_copy * delta_Y_copy'/1257;
    [V,D] = eig(Sigma_tilde);
    [D_descend, ind] = sort(diag(D),'descend');
    for j = 1:452
        N_k(j) = sum(diag(D)>D_descend(j));
        p(r,j) = (N_k(j)+1)/(R+1);
    end
end
% The p-values range from 0.0909 to 41.0909.