function build = cvxBuild()
% KNOCKOFFS.PRIVATE.CVXBUILD  Returns the build number of the installed CVX package.

% CVX provides no structured way to get this information,
% so we resort to a hack. However, the natural hack--parsing the output
% of 'cvx_version'--does not work, as decribed here:
%
%   http://ask.cvxr.com/question/3058
%
% The following quite terrible hack is reported to work for CVX versions
% 2 and 3 (and possibly even 1).

cvx_version(1);
global cvx___

fid = fopen([ cvx___.where, cvx___.fs, 'cvx_version.m' ]);
source = fread(fid, Inf, 'uint8=>char')';
fclose(fid);

buildStr = regexp(source, 'cvx_bld = ''(\d+)''', 'tokens');
build = str2double(buildStr{1});

end