A. The file, CLickLogSample.xls, contains 81 click-log records. The two queries are very similar. Each click record contains 4 fields: query id, search results, click pattern, and frequency. For example:

q2	[["u27","u38","u26","u39"],["u33","u44","u34","u43","u45","u36","u46"]]	[[1,2,3],[1,3,6]]	1

The record indicates that a user submitted query q2, and browsed two pages of search results. At the first page, he clicked the 1st, 2nd, and 3rd URLs. At the second page, he clicked 1st, 3rd, and 6th URLs. The frequency of the click pattern is 1. If the user skipped a page with no click, the click pattern will be empty "[]". 

B. The file, logdata-onequery.xls, contains 339 click-log records for a single query.

q	[1,2,3,4,5]	[["u1","u2","u3","u4","u5","u6","u7","u8","u9","u10"],["u9","u10","u11"],["u12","u13","u14"],["u15","u16","u17"],["u18","u19","u20"]]	[[9],[],[],[],[]]	1

Similar as above, with the only difference being the second column, which indicates the page number browsed. 

The data is provided by Dr. LI Hang and XU Jun from MSR-Asia and only used for research purpose.

--
Yuan Yao
School of Mathematical Sciences
Peking University
Beijing, 100871
Email: yuany@math.pku.edu.cn

April 2, 2010
