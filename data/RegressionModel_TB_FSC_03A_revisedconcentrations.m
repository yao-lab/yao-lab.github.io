% 2nd Order Regression Modeling with Fractional Factorial Design
%                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Only Parameters and Optimizaton parts needed to be modified
% 2. Follow the following rules before running the program:
%    1. For 2 or 3 level orthogonal array design only, not for higher
%    2. All drugs should have the same number of dosage levels
%    3. Orthogonal array: number of dosage levels between -1 and 0 should
%       be the same between 0 and 1 
%    4. Put data in the form of X1 in one column, X2 in second column,
%       then Y into the following column, in excel sheet
%    5. Put control data for normalization after all data points, in the 
%       same format as data, if data is already normalized, put a row of 1
%       after all data poitns
%    6. For higher dimension (> 3 drugs), modify line 157-198
%       [D1,D2,D3,D4,D5,...]=ndgrid(1:1:NOC);
%       X_All = [D1(:) D2(:) D3(:) D4(:) D5(:)...];
%       and also increase number of loops in all if statements:
%       if FitXOpt == ...
%    7. Check result: 1. correlation, 3. beta, 2. surface plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If want to minimize the performance, set MinMax = 0
% If want to maximize the performance, set MinMax = 1
MinMax = 1;

% If in_data is in orthogonal array level, and want to convert that to 
% discrete level for fitting,
%   set FitXOpt = 1
% If in_data is in orthogonal array level, and want to convert that to 
% actual concentration for fitting,
%   set FitXOpt = 2
% If in_data is in orthogonal array level, and want to directly use 
% orthogronal array level for fitting, choose this option only if you 
% have actual concentration levels in geometric series: {a,ar,ar^2,...}
%   set FitXOpt = 3
% If in_data is in not in orthogonal array level, but in its numbered
% dosage level, and want to directly use it for fitting,
%   set FitXOpt = 4
FitXOpt = 2;

% If you want to perform box-cox for the Y vector, set box_cox = 1, 
% if not,set box_cox = 0
box_cox = 0;

% Actual concentrations Table of each drug
% Make sure ConcTable are in Geometric series if you want to directly
% use orthogonal array level {-1,0,1} to fit
% Make sure number of concentration levels between - to 0 is the same 
% as the number of concentration levels between 0 to 1
ConcTable = [0 0.7000;
0 0.0780;
0 1.500;
0 0.0500;
0 0.0030;
0 0.0130;
0 0.0310;
0 0.0038;
0 0.0700;
0 0.0110;
0 4.00;
0 0.0008;
0 0.1000;
0 0.006];

   

% Corresponding dosage levels for each orthogonal array level
% e.g. According to ConcTable above, if corresponding 
% orthogonal array level is -1=0, 0=0.01, 1=10,
% then the OthArr2DiscLvl matrix is as follows
OthArr2DiscLvl = [1 2;            
                  1 2;  
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2;
                  1 2];

% Number of Test points, not including control(0,0,0)
Row = 128;
% Number of Drugs
Col = 14;

% Filename of your excel file
filename = 'TB-FSC-03A-data.xlsx';  %%
% Range of data in your excel file
xlrange = 'B2:P130'; % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Modeling Start Here %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% PREPARE DATA %%%%%%%%%%%%%%%
% Load data from spreadsheet
data = xlsread(filename,1,xlrange);
      
% Put data into matrix as follow
in_data  = data(1:Row,1:Col);
out_data = data(1:Row,Col+1);
% ctrl_in  = data(Row+1,1:Col);
% ctrl_out = data(Row+1,Col+1);

% Convert in_data from discrete level to actual concentration
if FitXOpt == 1
    % Convert in_data from orthogonal array level to discrete level
    for i=1:Row
        for j=1:Col
            if in_data(i,j)==-1 X_o(i,j)=OthArr2DiscLvl(j,1); end;
            if in_data(i,j)== 0 X_o(i,j)=OthArr2DiscLvl(j,2); end;
            if in_data(i,j)== 1 X_o(i,j)=OthArr2DiscLvl(j,size(OthArr2DiscLvl,2)); end;
        end
    end
elseif FitXOpt == 2   
   % Convert in_data from orthogonal array level to discrete level
    for i=1:Row
        for j=1:Col
            if in_data(i,j)==-1 X_o(i,j)=OthArr2DiscLvl(j,1); end;
            if in_data(i,j)== 0 X_o(i,j)=OthArr2DiscLvl(j,2); end;
            if in_data(i,j)== 1 X_o(i,j)=OthArr2DiscLvl(j,size(OthArr2DiscLvl,2)); end;
        end
    end
    % Convert X_o from discrete level to actual concentration
    for i=1:Row
    for j=1:Col                  
        X_o(i,j) = ConcTable(j,X_o(i,j)); 
    end
    end
elseif FitXOpt == 3
    X_o = in_data;
elseif FitXOpt == 4
    X_o = in_data;
else
    disp('Wrong FitXOpt Input. Read line 25,26.');
end

Y_o = out_data
% Perform boxcox transformation
% if box_cox == 0
% elseif box_cox == 1
%     [Y_o,lamba] = boxcox(Y_o);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% GENERATE MODEL %%%%%%%%%%%%%%%%

% Do not use regstats() if number of beta's is more than number of 
% equations, in that case, use regress(), LinearModel.fit is the best 
% among all
result = LinearModel.stepwise(X_o,Y_o,'linear', 'ResponseVar','Inhibition','PredictorVars',{'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14'})
beta = result.Coefficients.Estimate;
% Fitted value of Y
Y_o_rg = result.Fitted;
% Find correlation coefficient between Experimental Y and Modeled Y
R = corr(Y_o,Y_o_rg); % R is indeed the square root of result.MSE
str = ['Fitting Correlation is: ', num2str(R)]; disp(str);
% plot Residual plots
plotResiduals(result,'Fitted'); 


% Number of concentration levels for each drug
NOC = size(ConcTable,2);

    % Get complete set of input/output
    % meshgrid only for 3D, ndgrid for multi-dimension
    [D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14]=ndgrid(1:1:NOC);
    % Put into normal X form
    X_All = [D1(:) D2(:) D3(:) D4(:) D5(:) D6(:) D7(:) D8(:) D9(:) D10(:) D11(:) D12(:) D13(:) D14(:)];
    % Convert X_All from discrete level to actual concentration
    for i=1:size(X_All,1)
    for j=1:size(X_All,2)                                
        X_All(i,j) = ConcTable(j,X_All(i,j)); 
    end
    end


T=result.Formula.Terms;   
T=T(1:end,1:14);   
X_All_rg = x2fx(X_All,T);

% Calculate fitted Y value for all possible combinations
Y_All_rg=X_All_rg*beta;

% Find the optimal value
 
    [Opt_Y, ind] = max(Y_All_rg);
    Opt_Comb = X_All(ind,:);

str = ['Optimal Output of Y is ',num2str(Opt_Y)];
disp(str);
str1 = ['Optimal Combination of Y is '];
str = [str1,num2str(Opt_Comb)];
disp(str);


 ypred = predict(result,Opt_Comb)
 
%Manually find all combinations & outputs
All=[X_All Y_All_rg];
