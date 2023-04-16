JRM, 9/5/01

This directory contains the following three files.

1. crime2.dta
2. crime2.ssd01
3. crime2.xls

Essentially, these are all the same file.  The first is a (version 6)
STATA data set.  The second is a (version 6.12) SAS data set.  The
third is a (version 97/2000) Microsoft Excel spreadsheet.

The file is the same as Levitt's file #crime.dta#, but in addition
contains the variables #mayor#, #date_wa#, #date_my#, and #web#.
#mayor# is an indicator for mayoral election year.  #date_wa# is the
mayoral election date or the date as of which the next mayor will take
office, according to the World Almanac.  #date_my# (collected only for
some years) is the mayoral election date or the date as of which the
next mayor will take office, as reported in the Municipal Yearbook.
#web# is an indicator for whether I additionally verified mayoral
election dates for that city using the city web site.  Unfortunately,
I did not document the specific web site visited for each of these
verifications.  Of all the cities, Austin, TX, has easily the best web
site for data on mayoral elections.  See
http://malford.ci.austin.tx.us/election/search.cfm.
I have also attached variable labels, and (for the STATA version) a
data set label.  Otherwise, the data set is identical to Levitt's
#crime.dta#.

All analysis in "Do Electoral Cycles in Police Hiring Really Help Us
Estimate the Effect of Police on Crime? Comment" is based on
crime2.ssd01.  See the #programs# directory for the results and the
programs that generate them.


