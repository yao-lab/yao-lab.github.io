function [score,totalIncon,harmIncon]=Batch_Hodgerank(incomp, NodeNum)
%   Find the Hodge Decomposition of pairwise ranking data with four models:
%       model1: Uniform noise model, Y_hat(i,j) = 2 pij -1
%       model2: Bradley-Terry, Y_hat(i,j) = log(abs(pij+eps))-log(abs(1-pij-eps))
%       model3: Thurstone-Mosteller, Y_hat(i,j) ~ norminv(abs(1-pij-eps));
%       model4: Arcsin, Y_hat4(i,j) = asin(2*pij-1)
%
%   Input: 
%       incomp: n-by-2 matrix, the first column is the video with better quality 
%           and the second line is the video with worse quality.
%   Output:
%       score: 16-by-4 score matrix of 16 videos and 4 models
%       totalIncon: 4-by-1 total inconsistency
%       harmIncon: 4-by-1 harmonic inconsistency

%   by Qianqian Xu, Yuan Yao
%       CAS and Peking University
%       August, 2012
%
%   Reference:
%       Qianqian Xu, Qingming Huang, Tingting Jiang, Bowei Yan, Weisi Lin 
%   and Yuan Yao, "HodgeRank on Random Graphs for Subjective Video Quality  
%   Assessment", IEEE Transaction on Multimedia, vol. 14, no. 3, pp. 844-
%   857, 2012.

if ~isempty(incomp)
    count=zeros(NodeNum,NodeNum);
    eps=1e-4;
    sigma=1;

    for i = 1:size(incomp,1)
        for k = 0:NodeNum-1
            for l = 0:NodeNum-1
                if ((mod(incomp(i,1),NodeNum)==k) && (mod(incomp(i,2),NodeNum)==l))
                    if k==0,k=NodeNum;end
                    if l==0,l=NodeNum;end
                    count(k,l)=count(k,l)+1;
                end
            end
        end
    end
end

score_model1_rg=[]; % score of each item for model1
score_model2_rg=[]; % score of each item for model2
score_model3_rg=[]; % score of each item for model3
score_model4_rg=[]; % score of each item for model4
res_rg=[];  % total inconsistency
harmonic_res=[];
curl_res=[];

c0=count;
thresh=0;
G=((c0+c0')>thresh);

edges=[];
d0=[];
d1=[];
triangles=[];
k=0;
GG=[];

% Pairwise comparison skew-symmetric matrices
Y_hat1=zeros(NodeNum,NodeNum);
Y_hat2=zeros(NodeNum,NodeNum);
Y_hat3=zeros(NodeNum,NodeNum);
Y_hat4=zeros(NodeNum,NodeNum);

% Edge flow vectors
y_hat1=[];
y_hat2=[];
y_hat3=[];
y_hat4=[];
for j=1:(NodeNum-1)
    for i=(j+1):NodeNum
        if (G(i,j)>0)
            pij=c0(i,j)/(c0(i,j)+c0(j,i));
            % Uniform noise model
            Y_hat1(i,j)=2*pij-1;
            % Bradley-Terry model
            Y_hat2(i,j)=log(abs(pij+eps))-log(abs(1-pij-eps));
            % Thurstone-Mostelle model
            Y_hat3(i,j)=-sqrt(2)*sigma*norminv(abs(1-pij-eps));
            %arcsin model
            Y_hat4(i,j)=asin(2*pij-1);
            edges=[edges, [i;j]];
            k=k+1;
            GG(k)=c0(i,j)+c0(j,i);   %-------------------------------weight=number of raters
            %              GG(k)=1;  %-------------------------------unweighted
            y_hat1(k)=Y_hat1(i,j);
            y_hat2(k)=Y_hat2(i,j);
            y_hat3(k)=Y_hat3(i,j);
            y_hat4(k)=Y_hat4(i,j);

        end
    end
end

for i=1:NodeNum
    for j=(i+1):NodeNum
        for k=(j+1):NodeNum
            if ((G(i,j)>0)&&(G(j,k)>0)&&(G(k,i)>0))
                triangles = [triangles, [i,j,k]'];
            end
        end
    end
end
Y=[];

numEdge=size(edges,2);
numTriangle=size(triangles,2);

d0=zeros(numEdge,NodeNum);
for k=1:numEdge
    d0(k,edges(1,k))=1;
    d0(k,edges(2,k))=-1;
end

d1 = zeros(numTriangle,numEdge);
for k=1:numTriangle
    index=find(edges(1,:)==triangles(2,k) & edges(2,:)==triangles(1,k));
    d1(k,index)=1;
    index=find(edges(1,:)==triangles(3,k) & edges(2,:)==triangles(2,k));
    d1(k,index)=1;
    index=find(edges(1,:)==triangles(3,k) & edges(2,:)==triangles(1,k));%----------------------------edited by Yan on 11th, Mar
    d1(k,index)=-1;
end

% 0-degree Laplacian (graph Laplacian)
L0=[];
% d0_star is the conjugate of d0
d0_star = d0'*diag(GG);
L0=d0_star*d0;

% L1 Laplacian (upper part of 1-Laplacian)
% d1_star is the conjugate of d1
d1_star = diag(1./GG)*d1';
L1=d1*d1_star;


% Find divergence and global score for model I
div1 = d0_star*y_hat1';
x_global_m1=lsqr(L0,div1);
score_model1_rg=log(x_global_m1+1);

% Find curl and harmonic components for model II
curl1=d1*y_hat1';
curl_m1=lsqr(L1,curl1);
harmonic_m1=y_hat1'-d0*x_global_m1-d1_star*curl_m1;    % i.e. \hat{Y}^{(2)} in the paper ACM-MM11
res_rg(1,:) = (y_hat1'-d0*x_global_m1)'*diag(GG)*(y_hat1'-d0*x_global_m1)/(y_hat1*diag(GG)*y_hat1');
harmonic_res(1,:) = (harmonic_m1)'*diag(GG)*(harmonic_m1)/(y_hat1*diag(GG)*y_hat1');


div2 = d0_star*y_hat2';
x_global_m2=lsqr(L0,div2);
score_model2_rg=log(exp(x_global_m2));
curl2=d1*y_hat2';
curl_m2=lsqr(L1,curl2);
harmonic_m2=y_hat2'-d0*x_global_m2-d1_star*curl_m2;
res_rg(2,:) = (y_hat2'-d0*x_global_m2)'*diag(GG)*(y_hat2'-d0*x_global_m2)/(y_hat2*diag(GG)*y_hat2');
harmonic_res(2,:) = (harmonic_m2)'*diag(GG)*(harmonic_m2)/(y_hat2*diag(GG)*y_hat2');


div3 = d0_star*y_hat3';
x_global_m3=lsqr(L0,div3);
score_model3_rg=log(exp(x_global_m3));
curl3=d1*y_hat3';
curl_m3=lsqr(L1,curl3);
harmonic_m3=y_hat3'-d0*x_global_m3-d1_star*curl_m3;
res_rg(3,:) = (y_hat3'-d0*x_global_m3)'*diag(GG)*(y_hat3'-d0*x_global_m3)/(y_hat3*diag(GG)*y_hat3');
harmonic_res(3,:) = (harmonic_m3)'*diag(GG)*(harmonic_m3)/(y_hat3*diag(GG)*y_hat3');


div4 = d0_star*y_hat4';
x_global_m4=lsqr(L0,div4);
score_model4_rg=sin(x_global_m4);
curl4=d1*y_hat4';
curl_m4=lsqr(L1,curl4);
harmonic_m4=y_hat4'-d0*x_global_m4-d1_star*curl_m4;
res_rg(4,:) = (y_hat4'-d0*x_global_m4)'*diag(GG)*(y_hat4'-d0*x_global_m4)/(y_hat4*diag(GG)*y_hat4');
harmonic_res(4,:) = (harmonic_m4)'*diag(GG)*(harmonic_m4)/(y_hat4*diag(GG)*y_hat4');


score = [score_model1_rg,score_model2_rg,score_model3_rg,score_model4_rg];
totalIncon = res_rg;
harmIncon = harmonic_res; 
end


