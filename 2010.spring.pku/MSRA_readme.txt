The file contains 81 click-log records. The two queries are very similar. Each click record contains 4 fields: query id, search results, click pattern, and frequency. For example:

q2	[["u27","u38","u26","u39"],["u33","u44","u34","u43","u45","u36","u46"]]	[[1,2,3],[1,3,6]]	1

The record indicates that a user submitted query q2, and browsed two pages of search results. At the first page, he clicked the 1st, 2nd, and 3rd URLs. At the second page, he clicked 1st, 3rd, and 6th URLs. The frequency of the click pattern is 1. If the user skipped a page with no click, the click pattern will be empty "[]". 