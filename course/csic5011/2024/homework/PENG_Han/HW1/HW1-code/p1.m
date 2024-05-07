clc
x = load('train.0');
n = length(x(:,1));
p = length(x(1,:));

% a. Set up data matrix, pxn
x = x'; 

% b. Compute the sample mean µˆn and form
mu = mean(x, 2);
x_tilde = x - mu*ones(1,n);

%  c. Compute top k SVD
k = 20;
[U,S,V] = svd(x_tilde);

% d. plot eigenvalue curve
SIG = S*S';
lam = zeros(k,1);
tr = trace(SIG);
for i=1:k
    lam(i) = SIG(i,i);
end
figure(1)
plot(1:k,lam/tr)
xlabel('i')
ylabel('\lambda ratio')

% e.
figure(2)
imshow(reshape(mu,16,16))
title('Mean')
figure(3)
for i=1:k
    subplot(5,4,i)
    imshow(reshape(U(:,i),16,16))
end
sgtitle('Top-k principle components')

% f. 
v1 = V(:, 1); % Extract the first right singular vector
[~, sortedIndices] = sort(v1);% Get the indices that would sort v1 in ascending order
x_ordered = x(:, sortedIndices); % Order the images (xi) according to the sorted indices

% g.
coor = S*V';
figure(4)
plot(coor(1,:),coor(2,:),'.')
grid on

z = zeros(p,15);
co1 = [-7.83048,    -3.97638,   -0.0013141, 3.8958,     7.99848, -7.9792,       -4.22357,       -0.072283,   3.77624,   8.022,       -7.92829,  -4.13104,   -0.194194,  4.15276,    7.95296];
co2 = [4.37596,     3.90639,    3.96312,    3.73899,    4.19876, -0.0396296,    -0.0622869,     0.256878,   -0.285497,  0.411903,   -4.19335,    -3.78977,  -3.9798,    -4.03379,    -4.26289];
co = [co1;co2];
z = mu + U(:,1:2) * co;
%z(:,1) = mu -7.83048*U(:,1) + 4.37596*U(:,2);
%imshow(reshape(z(:,1),16,16))
figure(5)
for i=1:15
    subplot(3,5,i)
    imshow(reshape(z(:,i),16,16))
end
sgtitle('Image on the grid')



% rearrange the image in a matrix form
if 0
y=cell(n,1);
for i=1:n
    y{i}=reshape(x(i,:),16,16);
end
y1 = y(1:130,1);
y1 = reshape(y1,13,10);
A = cell(13,1);
for i=1:13
    for j=1:10
        A{i} = [A{i},y1{i,j}];
    end
end
B=[];
for i=1:13
    B=[B;A{i}];
end
imshow(B)
end

