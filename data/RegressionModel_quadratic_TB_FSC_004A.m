
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


% If you want to perform box-cox for the Y vector, set box_cox = 1, 
% if not,set box_cox = 0
box_cox = 0;

% Actual concentrations Table of each drug
% Make sure ConcTable are in Geometric series if you want to directly
% use orthogonal array level {-1,0,1} to fit
% Make sure number of concentration levels between - to 0 is the same 
% as the number of concentration levels between 0 to 1
ConcTable = [   0	0.052	0.1040 ; %D2
                0	0.12	0.2400 ; %D4
                0	0.0025	0.0050 ; %D8
                0	0.025	0.0500 ; %D9
                0	0.0135	0.0270 ; %D10
                0	7.5	   15.0000 ; %D11
                0	0.0005	0.0010 ; %D12
                0	0.125	0.2500 ; %D13
                0	0.00425	0.0085 ]; %D14
         

% Corresponding dosage levels for each orthogonal array level
% e.g. According to ConcTable above, if corresponding 
% orthogonal array level is -1=0, 0=0.01, 1=10,
% then the OthArr2DiscLvl matrix is as follows
OthArr2DiscLvl = [1 2 3; %D2            
                  1 2 3; %D4
                  1 2 3; %D8
                  1 2 3; %D9
                  1 2 3; %D10
                  1 2 3; %D11
                  1 2 3; %D12
                  1 2 3; %D13
                  1 2 3 ]; %D14
            

% Number of Test points, not including control(0,0,0)
Row = 155;
% Number of Drugs
Col = 9;

% Filename of your excel file
filename = 'TB-FSC-004A-data.xlsx';
% Range of data in your excel file
xlrange = 'B3:K158'; % Including control
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Modeling Start Here %%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% PREPARE DATA %%%%%%%%%%%%%%%
% Load data from spreadsheet
data = xlsread(filename,1,xlrange);
      
% Put data into matrix as follow
in_data  = data(1:Row,1:Col);
out_data = data(1:Row,Col+1);
ctrl_in  = data(Row+1,1:Col);



 
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



Y_o = out_data;
% Perform boxcox transformation
if box_cox == 0
elseif box_cox == 1
    [Y_o,lamba] = boxcox(Y_o);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% GENERATE MODEL %%%%%%%%%%%%%%%%

% Do not use regstats() if number of beta's is more than number of 
% equations, in that case, use regress(), LinearModel.fit is the best 
% among all
result = LinearModel.stepwise(X_o,Y_o,'quadratic', 'ResponseVar','Inhibition','PredictorVars',{'D2','D4','D8','D9','D10','D11','D12','D13','D14'})
beta = result.Coefficients.Estimate;
% Fitted value of Y
Y_o_rg = result.Fitted;
% Find correlation coefficient between Experimental Y and Modeled Y
R = corr(Y_o,Y_o_rg); % R is indeed the square root of result.MSE
str = ['Fitting Correlation is: ', num2str(R)]; disp(str);
% plot Residual plots
plotResiduals(result,'Fitted'); 

% Plot 
figure('Name','Result Stepwise - No boxcox')
subplot(2,2,1)
plotResiduals(result,'Fitted');
subplot(2,2,2)
plotDiagnostics(result,'cookd');
subplot(2,2,3)
plotResiduals(result,'probability')
subplot(2,2,4)
plotResiduals(result,'histogram')

plotSlice(result);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% OPTIMIZATION %%%%%%%%%%%%

% Number of concentration levels for each drug
NOC = size(ConcTable,2);

    % Get complete set of input/output
    % meshgrid only for 3D, ndgrid for multi-dimension
    [D2,D4,D5,D8,D9,D10,D11,D12,D13,D14]=ndgrid(1:1:NOC);
    % Put into normal X form
    X_All = [D2(:) D4(:) D8(:) D9(:) D10(:) D11(:) D12(:) D13(:) D14(:)];
    % Convert X_All from discrete level to actual concentration
    for i=1:size(X_All,1)
    for j=1:size(X_All,2)                                
        X_All(i,j) = ConcTable(j,X_All(i,j)); 
    end
    end


save RegressionModel_quadratic_TB_FSC_004A.mat