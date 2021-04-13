% KNOCKOFF The Model-Free Knockoff Filter for controlling the false discovery rate (FDR).
%
% The knockoffs framework constructs artificial 'knockoffs' for the 
% variables in a statistical model and then selects only variables that 
% are clearly better than their fake copies. Our model-free approach makes 
% knockoffs possible for data from any model, no matter how high-dimensional.
%
% Reference: http://statweb.stanford.edu/~candes/MF_Knockoffs/
%
% Files
%   create    - Create Model-Free knockoffs given the model parameters and a 
%   filter    - Run the knockoff filter on a data set.
%   select    - Select variables based on the knockoff statistics
%   threshold - Compute the threshold for variable selection
%
% For more information, try typing:
%   - help knockoffs.knocks
%   - help knockoffs.stats
%
% For usage examples, see the scripts in the 'examples' directory