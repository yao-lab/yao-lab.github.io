% Wishart matrix
% 	S = 1/n*X*X.', X is p-by-n, X ij i.i.d N(0,1),
% Eigenvalue distribution of S converges to Marcenko-Pastur distribution with parameter gamma = p/n

gamma = 2; % if gamma>1, there will be a spike in MP distribution at 0
a = (1-sqrt(gamma))^2;
b = (1+sqrt(gamma))^2;

f_MP = @(t) sqrt(max(b-t, 0).*max(t-a, 0) )./(2*pi*gamma*t); %MP Distribution

%non-zero eigenvalue part
n = 400;
p = n*gamma;
X = randn(p,n);
S = 1/n*(X*X.');
evals = sort( eig(S), 'descend');
nbin = 100;
[nout, xout] = hist(evals, nbin);
hx = xout(2) - xout(1); % step size, used to compute frequency below
x1 = evals(end) -1;
x2 = evals(1) + 1; % two end points
xx = x1+hx/2: hx: x2;
fre = f_MP(xx)*hx;
figure,
h = bar(xout, nout/p);
set(h, 'BarWidth', 1, 'FaceColor', 'w', 'EdgeColor', 'b');
hold on;
plot(xx, fre, '--r');

if gamma > 1 % there are (1-1/gamma)*p zero eigenvalues
    axis([-1 x2+1 0 max(fre)*2]);
end