# Split Knockoffs

This is the package for split Knockoffs. For usage of this package, check "Contents.m" in "+split_knockoffs" folder.

This package is built based on the "+knockoffs" folder from [Knockoffs for matlab](https://web.stanford.edu/group/candes/knockoffs/software/knockoffs/). The preliminary for this package is [glmnet in matlab](https://web.stanford.edu/~hastie/glmnet_matlab/). A fortran compiler is required for using glmnet in matlab. You may download the [Intel Fortran Compiler](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html#fortran), and follow the [guideline](https://www.mathworks.com/support/requirements/previous-releases.html) to install the fortran compiler with [Visual Studio](https://visualstudio.microsoft.com/).

To reproduce the figures and tables in the paper, please go to "simulation" folder and "AD_experiments" folder and check repective "Contents.m" file. The recommended and default setting for the range of lambda on reproducing the results is 10<sup>0</sup> - 10<sup>-6</sup> with step size 0.01. 